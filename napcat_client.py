"""
NapCat 协议接入模块 (NapCat WebSocket Client)

该模块负责维护与 NapCatQQ 的正向 WebSocket 连接，
接收消息并通过回调机制向外暴露，同时提供消息发送接口。
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol

# 配置日志
logger = logging.getLogger(__name__)


class NapCatConfigError(Exception):
    """NapCat 配置相关异常"""
    pass


class NapCatConnectionError(Exception):
    """NapCat 连接相关异常"""
    pass


class MessageCallback:
    """消息回调的类型提示类"""
    __call__: Callable[[dict], Any]


class NapCatClient:
    """
    NapCat WebSocket 客户端
    
    负责与 NapCat 建立连接、接收消息、发送消息，以及断线重连。
    不包含任何业务逻辑，通过回调函数将消息传递给外部处理。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 NapCat 客户端
        
        Args:
            config_path: 配置文件路径，默认为同目录下的 napcat_config.json
        """
        self._config: dict = {}
        self._config_path = config_path
        self._ws: Optional[WebSocketClientProtocol] = None
        self._message_callback: Optional[Callable[[dict], Any]] = None
        self._running: bool = False
        self._connected: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 加载配置
        self._load_config()
    
    def _resolve_env_vars(self, value: str) -> str:
        """
        解析字符串中的环境变量占位符
        
        支持格式：${ENV_VAR_NAME} 或 ${ENV_VAR_NAME:default_value}
        """
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) or ''
            return os.environ.get(var_name, default_value)
        
        return re.sub(pattern, replace_env_var, value)
    
    def _resolve_config_values(self, raw_config: Any) -> Any:
        """递归解析配置中所有值的环境变量"""
        if isinstance(raw_config, dict):
            return {k: self._resolve_config_values(v) for k, v in raw_config.items()}
        elif isinstance(raw_config, list):
            return [self._resolve_config_values(item) for item in raw_config]
        elif isinstance(raw_config, str):
            return self._resolve_env_vars(raw_config)
        else:
            return raw_config
    
    def _load_config(self) -> None:
        """
        加载配置文件
        
        如果配置文件不存在或字段缺失，使用默认值。
        """
        # 默认配置
        default_config = {
            'napcat': {
                'host': '127.0.0.1',
                'port': '3001',
                'access_token': '',
                'use_wss': False
            },
            'reconnect': {
                'enabled': True,
                'interval': 3,
                'max_retries': -1  # -1 表示无限重试
            },
            'heartbeat': {
                'enabled': True,
                'interval': 30
            }
        }
        
        # 确定配置文件路径
        if self._config_path:
            path = Path(self._config_path)
        else:
            path = Path(__file__).parent / 'napcat_config.json'
        
        # 尝试加载配置文件
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                
                # 解析环境变量
                loaded_config = self._resolve_config_values(raw_config)
                
                # 合并配置（用户配置覆盖默认值）
                for section in default_config:
                    if section in loaded_config:
                        default_config[section].update(loaded_config[section])
                
                logger.info(f"配置文件加载成功: {path}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"配置文件格式错误，使用默认配置: {e}")
            except Exception as e:
                logger.warning(f"配置文件读取失败，使用默认配置: {e}")
        else:
            logger.warning(f"配置文件不存在，使用默认配置: {path}")
        
        self._config = default_config
        
        # 确保端口为整数
        self._config['napcat']['port'] = int(self._config['napcat']['port'])
    
    def _build_ws_url(self) -> str:
        """
        构建 WebSocket 连接 URL
        
        根据 NapCat 规范，access_token 可以放在 URL 参数或请求头中。
        这里使用 URL 参数方式。
        
        Returns:
            完整的 WebSocket URL
        """
        napcat = self._config['napcat']
        host = napcat['host']
        port = napcat['port']
        use_wss = napcat.get('use_wss', False)
        access_token = napcat.get('access_token', '')
        
        protocol = 'wss' if use_wss else 'ws'
        url = f"{protocol}://{host}:{port}"
        
        # 如果配置了 access_token，添加到 URL 参数
        if access_token:
            url += f"?access_token={access_token}"
        
        return url
    
    @property
    def is_connected(self) -> bool:
        """返回当前连接状态"""
        return self._connected and self._ws is not None and self._ws.open
    
    async def _heartbeat_loop(self) -> None:
        """
        心跳保活循环
        
        定期发送心跳包以保持连接活跃。
        NapCat 的正向 WS 不严格要求心跳，但建议定期发送。
        """
        heartbeat_config = self._config.get('heartbeat', {})
        interval = heartbeat_config.get('interval', 30)
        
        while self._running and self._connected:
            try:
                await asyncio.sleep(interval)
                
                if self.is_connected:
                    # 发送心跳包（NapCat 使用空的心跳事件或 ping）
                    # 这里发送一个空的 API 调用来保持连接
                    try:
                        await self._ws.ping()
                        logger.debug("心跳包发送成功")
                    except Exception as e:
                        logger.warning(f"心跳包发送失败: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳循环异常: {e}")
    
    async def _message_handler(self, raw_message: str) -> None:
        """
        处理接收到的原始消息
        
        解析 JSON 并触发外部回调函数。
        
        Args:
            raw_message: WebSocket 接收到的原始字符串消息
        """
        try:
            # 解析 JSON
            event_data = json.loads(raw_message)
            
            # 记录接收到的消息（调试用）
            post_type = event_data.get('post_type', 'unknown')
            logger.debug(f"收到消息 [type={post_type}]: {raw_message[:200]}")
            
            # 触发外部回调（在独立任务中执行，避免阻塞消息接收循环）
            if self._message_callback:
                try:
                    # 支持同步和异步回调，在后台任务中执行
                    result = self._message_callback(event_data)
                    if asyncio.iscoroutine(result):
                        # 创建独立任务执行，不阻塞当前循环
                        asyncio.create_task(result)
                except Exception as e:
                    logger.error(f"消息回调执行异常: {e}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"消息 JSON 解析失败: {e}, 原始消息: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"消息处理异常: {e}")
    
    async def _connect_and_listen(self) -> None:
        """
        建立连接并监听消息
        
        这是核心事件循环，负责建立连接、接收消息和处理断线。
        """
        url = self._build_ws_url()
        reconnect_config = self._config.get('reconnect', {})
        reconnect_enabled = reconnect_config.get('enabled', True)
        reconnect_interval = reconnect_config.get('interval', 3)
        max_retries = reconnect_config.get('max_retries', -1)
        
        retry_count = 0
        
        while self._running:
            try:
                logger.info(f"正在连接 NapCat: {url.split('?')[0]}...")  # 隐藏 token
                
                # 建立 WebSocket 连接
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    retry_count = 0  # 重置重试计数
                    
                    logger.info("NapCat WebSocket 连接成功！")
                    
                    # 启动心跳任务
                    if self._config.get('heartbeat', {}).get('enabled', True):
                        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    
                    # 消息接收循环
                    async for message in ws:
                        if not self._running:
                            break
                        await self._message_handler(message)
                
            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket 连接关闭: code={e.code}, reason={e.reason}")
                
            except websockets.InvalidStatusCode as e:
                logger.error(f"WebSocket 连接失败: HTTP {e.status_code}")
                
            except ConnectionRefusedError:
                logger.error("连接被拒绝，请检查 NapCat 是否已启动")
                
            except Exception as e:
                logger.error(f"WebSocket 连接异常: {type(e).__name__}: {e}")
            
            finally:
                # 清理连接状态
                self._connected = False
                self._ws = None
                
                # 取消心跳任务
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    self._heartbeat_task = None
            
            # 断线重连逻辑
            if not self._running:
                break
                
            if reconnect_enabled:
                retry_count += 1
                
                # 检查是否达到最大重试次数（-1 表示无限）
                if max_retries > 0 and retry_count > max_retries:
                    logger.error(f"已达到最大重试次数 ({max_retries})，停止重连")
                    break
                
                logger.info(f"{reconnect_interval} 秒后尝试重连... (第 {retry_count} 次)")
                await asyncio.sleep(reconnect_interval)
            else:
                logger.info("自动重连已禁用，停止连接")
                break
    
    async def start(self, message_callback: Callable[[dict], Any]) -> None:
        """
        启动 NapCat 客户端
        
        这是模块的主入口，传入消息回调函数后开始连接和监听。
        该方法会阻塞，通常应该在独立的任务中运行。
        
        Args:
            message_callback: 异步消息回调函数，签名为 async def callback(event_data: dict)
            
        Example:
            >>> async def on_message(event):
            ...     print(f"收到事件: {event}")
            >>> 
            >>> client = NapCatClient()
            >>> await client.start(on_message)
        """
        if self._running:
            logger.warning("客户端已在运行中")
            return
        
        self._message_callback = message_callback
        self._running = True
        
        logger.info("NapCat 客户端启动...")
        
        try:
            await self._connect_and_listen()
        finally:
            self._running = False
            logger.info("NapCat 客户端已停止")
    
    async def stop(self) -> None:
        """
        停止 NapCat 客户端
        
        优雅地关闭连接并停止所有任务。
        """
        logger.info("正在停止 NapCat 客户端...")
        self._running = False
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"关闭 WebSocket 时出错: {e}")
    
    async def send_group_msg(self, group_id: int, message: str) -> dict:
        """
        发送群消息
        
        按照 NapCat 的 API 格式发送消息。
        
        Args:
            group_id: 目标群号
            message: 消息内容（支持纯文本或 CQ 码）
            
        Returns:
            API 响应结果
            
        Raises:
            NapCatConnectionError: 连接未建立时抛出
        """
        if not self.is_connected:
            raise NapCatConnectionError("WebSocket 连接未建立，无法发送消息")
        
        # 构建 NapCat API 请求格式
        payload = {
            "action": "send_group_msg",
            "params": {
                "group_id": group_id,
                "message": message
            }
        }
        
        try:
            await self._ws.send(json.dumps(payload))
            logger.debug(f"已发送群消息: group_id={group_id}, message={message[:50]}...")
            
            # 注意：这里不等待响应，NapCat 的响应会通过消息流返回
            # 如果需要获取发送结果，可以在回调中处理 meta_event 类型的响应
            return {"status": "sent", "group_id": group_id}
            
        except Exception as e:
            logger.error(f"发送群消息失败: {e}")
            raise NapCatConnectionError(f"发送消息失败: {e}")
    
    async def send_private_msg(self, user_id: int, message: str) -> dict:
        """
        发送私聊消息
        
        Args:
            user_id: 目标用户 QQ 号
            message: 消息内容
            
        Returns:
            API 响应结果
        """
        if not self.is_connected:
            raise NapCatConnectionError("WebSocket 连接未建立，无法发送消息")
        
        payload = {
            "action": "send_private_msg",
            "params": {
                "user_id": user_id,
                "message": message
            }
        }
        
        try:
            await self._ws.send(json.dumps(payload))
            logger.debug(f"已发送私聊消息: user_id={user_id}")
            return {"status": "sent", "user_id": user_id}
            
        except Exception as e:
            logger.error(f"发送私聊消息失败: {e}")
            raise NapCatConnectionError(f"发送消息失败: {e}")
    
    async def call_api(self, action: str, params: dict) -> dict:
        """
        通用 API 调用接口
        
        用于调用 NapCat 支持的任意 API。
        
        Args:
            action: API 动作名称（如 get_login_info, get_group_list 等）
            params: API 参数
            
        Returns:
            发送状态
        """
        if not self.is_connected:
            raise NapCatConnectionError("WebSocket 连接未建立")
        
        payload = {
            "action": action,
            "params": params
        }
        
        try:
            await self._ws.send(json.dumps(payload))
            logger.debug(f"已调用 API: {action}")
            return {"status": "sent", "action": action}
            
        except Exception as e:
            logger.error(f"API 调用失败: {action}, {e}")
            raise NapCatConnectionError(f"API 调用失败: {e}")


# ============================================================
# 模块级便捷函数
# ============================================================

_client_instance: Optional[NapCatClient] = None


def get_client(config_path: Optional[str] = None) -> NapCatClient:
    """
    获取全局 NapCat 客户端实例
    
    Args:
        config_path: 配置文件路径（仅首次调用时有效）
        
    Returns:
        NapCatClient 实例
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = NapCatClient(config_path)
    return _client_instance

# ============================================================
# 模块级便捷函数
# ============================================================

_client_instance: Optional[NapCatClient] = None

def get_client(config_path: Optional[str] = None) -> NapCatClient:
    """获取全局 NapCat 客户端实例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = NapCatClient(config_path)
    return _client_instance

# ============================================================
# 测试代码 (正确封顶版)
# ============================================================

async def _test_napcat_client():
    """测试函数"""
    # 1. 设置极其简单的日志，方便你在控制台看清动作
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 2. 定义一个模拟的外部大脑回调
    async def dummy_on_message(event):
        post_type = event.get('post_type')
        if post_type == 'message':
            print(f"👉 拦截到群/私聊消息: {event.get('raw_message')}")
        elif post_type == 'meta_event':
            pass # 忽略无聊的心跳事件

    # 3. 实例化并启动
    client = get_client()
    print("=" * 50)
    print("🛠️ NapCat 客户端独立测试启动")
    print("=" * 50)
    
    try:
        await client.start(dummy_on_message)
    except asyncio.CancelledError:
        print("\n正在优雅停止...")
    finally:
        await client.stop()

if __name__ == "__main__":
    try:
        asyncio.run(_test_napcat_client())
    except KeyboardInterrupt:
        print("测试手动终止。")
