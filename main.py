"""
LessBot 核心调度中枢 (Core Dispatcher)

该模块是整个机器人的大脑，负责：
1. 消息分发与处理器路由（责任链模式）
2. 群聊消息的防抖缓冲（滑动窗口）
3. 双模型管线调用（导演-演员架构）
4. 物理层模拟（真人化回复节奏）
5. 配置驱动的外部化参数管理
"""

import asyncio
import json
import logging
import random
import time
import re
import base64
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from napcat_client import get_client, NapCatClient
from llm_caller import ask_llm

# ============================================================
# 日志配置
# ============================================================

logger = logging.getLogger("LessBot")


# ============================================================
# 配置管理
# ============================================================

@dataclass
class BotConfig:
    """
    机器人配置数据类
    
    封装所有可配置的行为参数，支持从 JSON 文件加载。
    """
    # 机器人元信息
    name: str = "LessBot"
    version: str = "1.0.0"
    
    # 行为参数
    debounce_seconds: float = 3.0
    typing_delay_min: float = 0.8
    typing_delay_max: float = 1.5
    max_context_messages: int = 50
    reply_probability: float = 0.5
    allowed_groups: List[int] = field(default_factory=list)
    
    # 模型配置
    director_model: str = "deepseek-reasoner"
    actor_model: str = "deepseek-chat"
    vision_model: str = "qwen3.5-plus"
    
    # 日志级别
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> 'BotConfig':
        """
        从 JSON 文件加载配置
        
        Args:
            config_path: 配置文件路径，默认为同目录下的 main_config.json
            
        Returns:
            BotConfig 实例
        """
        # 确定配置文件路径
        if config_path:
            path = Path(config_path)
        else:
            path = Path(__file__).parent / 'main_config.json'
        
        # 如果配置文件不存在，使用默认值
        if not path.exists():
            logger.warning(f"配置文件不存在: {path}，使用默认配置")
            return cls()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            # 解析配置
            bot_info = raw_config.get('bot', {})
            behavior = raw_config.get('behavior', {})
            models = raw_config.get('models', {})
            log_cfg = raw_config.get('logging', {})
            
            return cls(
                name=bot_info.get('name', 'LessBot'),
                version=bot_info.get('version', '1.0.0'),
                debounce_seconds=float(behavior.get('debounce_seconds', 3.0)),
                typing_delay_min=float(behavior.get('typing_delay_min', 0.8)),
                typing_delay_max=float(behavior.get('typing_delay_max', 1.5)),
                max_context_messages=int(behavior.get('max_context_messages', 50)),
                reply_probability=float(behavior.get('reply_probability', 0.5)),
                allowed_groups=behavior.get('allowed_groups', []),
                director_model=models.get('director', 'deepseek-reasoner'),
                actor_model=models.get('actor', 'deepseek-chat'),
                vision_model=models.get('vision', 'qwen3.5-plus'),
                log_level=log_cfg.get('level', 'INFO'),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"配置文件 JSON 解析失败: {e}")
            return cls()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return cls()
    
    def setup_logging(self) -> None:
        """根据配置设置日志"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S',
            force=True  # 强制重新配置
        )


# ============================================================
# 数据模型定义
# ============================================================

@dataclass
class GroupMessage:
    """群消息数据结构"""
    group_id: int
    user_id: int
    raw_message: str
    sender_name: str = ""
    timestamp: float = 0.0
    is_at_me: bool = False
    
    def format_context(self) -> str:
        """格式化为上下文字符串"""
        name = self.sender_name or str(self.user_id)
        return f"[{name}]: {self.raw_message}"


@dataclass
class MessageBuffer:
    """
    消息缓冲池（用于防抖）
    
    以 group_id 为维度，累积消息并管理定时器。
    """
    messages: List[GroupMessage] = field(default_factory=list)
    timer_task: Optional[asyncio.Task] = None


# ============================================================
# 处理器抽象基类（责任链模式）
# ============================================================

class MessageHandler(ABC):
    """
    消息处理器抽象基类
    
    所有业务处理器都应继承此类并实现 handle 方法。
    处理器可以决定是否处理消息，以及是否继续传递给下一个处理器。
    """
    
    def __init__(self, next_handler: Optional['MessageHandler'] = None):
        """
        初始化处理器
        
        Args:
            next_handler: 责任链中的下一个处理器
        """
        self._next_handler = next_handler
    
    def set_next(self, handler: 'MessageHandler') -> 'MessageHandler':
        """
        设置下一个处理器（便于链式调用）
        
        Args:
            handler: 下一个处理器实例
            
        Returns:
            传入的处理器实例，便于链式构建
        """
        self._next_handler = handler
        return handler
    
    @abstractmethod
    async def can_handle(self, event: dict) -> bool:
        """
        判断是否可以处理该事件
        
        Args:
            event: 原始事件数据
            
        Returns:
            True 表示可以处理，False 表示跳过
        """
        pass
    
    @abstractmethod
    async def handle(self, event: dict, client: NapCatClient) -> bool:
        """
        处理消息事件
        
        Args:
            event: 原始事件数据
            client: NapCat 客户端实例，用于发送消息
            
        Returns:
            True 表示已处理，不需要继续传递；
            False 表示未处理或需要继续传递
        """
        pass
    
    async def process(self, event: dict, client: NapCatClient) -> bool:
        """
        责任链处理入口
        
        按照责任链顺序依次尝试处理，直到有处理器接收。
        
        Args:
            event: 原始事件数据
            client: NapCat 客户端实例
            
        Returns:
            是否被任一处理器处理
        """
        # 检查当前处理器是否可以处理
        if await self.can_handle(event):
            handled = await self.handle(event, client)
            if handled:
                return True
        
        # 传递给下一个处理器
        if self._next_handler:
            return await self._next_handler.process(event, client)
        
        return False

# ============================================================
# 视觉处理中枢 (Vision Analyzer)
# ============================================================

class VisionAnalyzer:
    """
    视觉中枢 (Base64 版)
    """
    @staticmethod
    async def analyze(image_url: str, model_name: str) -> str:
        try:
            logger.info(f"👁️ 正在调用视觉模型 ({model_name}) 分析图片...")
            
            # 1. 清洗 URL（修复 CQ 码的 &amp; 转义 Bug）
            clean_url = image_url.replace('&amp;', '&')
            
            # 2. 【核心绝招】本地下载并转为 Base64，彻底破解腾讯防盗链
            try:
                async with httpx.AsyncClient() as client:
                    img_resp = await client.get(clean_url, timeout=5.0)
                    if img_resp.status_code == 200:
                        b64_data = base64.b64encode(img_resp.content).decode('utf-8')
                        clean_url = f"data:image/jpeg;base64,{b64_data}"
                        logger.info("✅ 图片已成功转为 Base64 内存直传模式")
                    else:
                        logger.warning(f"图片下载失败 (状态码 {img_resp.status_code})，尝试回退到 URL 模式")
            except Exception as dl_e:
                logger.warning(f"图片下载异常，尝试回退到 URL 模式: {dl_e}")
            
            # 3. 构造极简视觉提示词
            vision_prompt = "你是一个群聊视觉助手。请简短描述这张图片的核心内容。如果是表情包请指出它的情绪，如果是梗图请解释，如果是文字截图请提取核心意思。字数严格控制在10-50字以内。"
            
            # 4. 调用大模型
            description = await ask_llm(
                model_name=model_name,
                prompt_content=vision_prompt,
                image_url=clean_url
            )
            
            # 5. 拦截 llm_caller 抛上来的错误字符串，防止污染聊天记忆
            if description.startswith("[LLM"):
                logger.error(f"❌ 视觉 API 拒绝了请求: {description}")
                return "一张无法显示的图片"
                
            return description.strip()
            
        except Exception as e:
            logger.error(f"视觉分析发生未知异常: {e}")
            return "无法看清的图片"

# ============================================================
# 群聊 LLM 处理器（核心业务逻辑）
# ============================================================

class GroupLLMHandler(MessageHandler):
    """
    群聊 LLM 处理器
    
    实现核心业务逻辑：
    1. 滑动窗口防抖
    2. 导演模型（状态生成）
    3. 演员模型（回复生成）
    4. 物理层模拟（真人化发送）
    
    所有参数通过构造函数注入，支持外部配置驱动。
    """
    
    def __init__(
        self,
        next_handler: Optional[MessageHandler] = None,
        debounce_seconds: float = 3.0,
        typing_delay_min: float = 0.8,
        typing_delay_max: float = 1.5,
        max_context_messages: int = 50,
        director_model: str = "deepseek-reasoner",
        actor_model: str = "deepseek-chat",
        vision_model: str = "qwen3.5-plus",
        allowed_groups: List[int] = None,
        reply_probability: float = 0.5
    ):
        """
        初始化群聊 LLM 处理器
        
        Args:
            next_handler: 责任链中的下一个处理器
            debounce_seconds: 防抖时间窗口（秒）
            typing_delay_min: 打字延迟下限（秒）
            typing_delay_max: 打字延迟上限（秒）
            max_context_messages: 最大上下文消息数
            director_model: 导演模型名称（配置文件中的 key）
            actor_model: 演员模型名称（配置文件中的 key）
        """
        super().__init__(next_handler)
        
        # 【配置注入】所有参数从外部传入，不再硬编码
        self._debounce_seconds = debounce_seconds
        self._typing_delay_min = typing_delay_min
        self._typing_delay_max = typing_delay_max
        self._max_context_messages = max_context_messages
        self._director_model = director_model
        self._actor_model = actor_model
        self._vision_model = vision_model
        self._allowed_groups = allowed_groups or []
        self._reply_probability = reply_probability
        
        # 消息缓冲池字典：{group_id: MessageBuffer}
        self._buffers: Dict[int, MessageBuffer] = {}
        
        # 防抖锁，防止并发操作同一群的缓冲区
        self._buffer_locks: Dict[int, asyncio.Lock] = {}
        
        logger.info(
            f"GroupLLMHandler 初始化完成: "
            f"防抖={debounce_seconds}s, "
            f"打字延迟={typing_delay_min}~{typing_delay_max}s, "
            f"导演={director_model}, 演员={actor_model}"
        )
    
    @classmethod
    def from_config(
        cls,
        config: BotConfig,
        next_handler: Optional[MessageHandler] = None
    ) -> 'GroupLLMHandler':
        """
        从配置对象创建处理器实例
        
        Args:
            config: BotConfig 配置对象
            next_handler: 责任链中的下一个处理器
            
        Returns:
            配置化的 GroupLLMHandler 实例
        """
        return cls(
            next_handler=next_handler,
            debounce_seconds=config.debounce_seconds,
            typing_delay_min=config.typing_delay_min,
            typing_delay_max=config.typing_delay_max,
            max_context_messages=config.max_context_messages,
            director_model=config.director_model,
            actor_model=config.actor_model,
            vision_model=config.vision_model,
            allowed_groups=config.allowed_groups,
            reply_probability=config.reply_probability
        )
    
    def _get_lock(self, group_id: int) -> asyncio.Lock:
        """获取指定群的缓冲区锁"""
        if group_id not in self._buffer_locks:
            self._buffer_locks[group_id] = asyncio.Lock()
        return self._buffer_locks[group_id]
    
    async def can_handle(self, event: dict) -> bool:
        """
        判断是否为群聊文本消息
        
        只处理群聊消息，且消息类型为文本。
        """
        # 检查是否为消息事件
        if event.get('post_type') != 'message':
            return False
        
        # 检查是否为群消息
        if event.get('message_type') != 'group':
            return False
        
        # 检查是否有文本内容
        raw_message = event.get('raw_message', '')
        if not raw_message or not raw_message.strip():
            return False
        
        group_id = event.get('group_id')
        if self._allowed_groups and group_id not in self._allowed_groups:
            return False
        
        return True
    
    async def handle(self, event: dict, client: NapCatClient) -> bool:
        """
        处理群聊消息
        
        将消息加入缓冲池并启动/重置防抖定时器。
        """
        group_id = event.get('group_id')
        user_id = event.get('user_id')
        raw_message = event.get('raw_message', '')
        self_id = event.get('self_id', 0)
        
        # 获取发送者信息
        sender = event.get('sender', {})
        sender_name = sender.get('card') or sender.get('nickname', '')
        
        is_at_me = f"[CQ:at,qq={self_id}]" in raw_message

        # 🎯 【新增：图像拦截与视觉解析】
        # 使用正则提取 NapCat CQ 码中的图片 URL: [CQ:image,file=...,url=https://...]
        image_urls = re.findall(r'\[CQ:image,.*?url=([^\]]+)\]', raw_message)
        # 把长长的 CQ 码替换成干净的占位符
        clean_message = re.sub(r'\[CQ:image.*?\]', '[图片]', raw_message)

        if image_urls:
            logger.info(f"[群{group_id}] 拦截到 {len(image_urls)} 张图片，启动视觉解析...")
            descriptions = []
            for url in image_urls:
                # 阻塞式获取图片文本描述（在防抖入池前完成翻译）
                desc = await VisionAnalyzer.analyze(url, self._vision_model)
                descriptions.append(desc)
            
            # 将视觉翻译结果悄悄附加在消息文字后面
            clean_message += f" (视觉系统提示：{', '.join(descriptions)})"

        # 构建消息对象
        msg = GroupMessage(
            group_id=group_id,
            user_id=user_id,
            raw_message=clean_message,
            sender_name=sender_name,
            timestamp=time.time(),
            is_at_me=is_at_me
        )
        
        # 将消息加入缓冲池（带防抖）
        await self._add_to_buffer(msg, client)
        
        return True  # 已处理，不传递给下一个处理器
    
    async def _add_to_buffer(self, msg: GroupMessage, client: NapCatClient) -> None:
        """
        将消息加入缓冲池并重置防抖定时器
        
        【核心防抖逻辑】
        1. 获取该群的缓冲区锁
        2. 将消息追加到缓冲池
        3. 取消旧的定时器（如果有）
        4. 创建新的防抖定时器
        5. 如果防抖期内无新消息，触发处理
        """
        group_id = msg.group_id
        lock = self._get_lock(group_id)
        
        async with lock:
            # 初始化缓冲池（如果不存在）
            if group_id not in self._buffers:
                self._buffers[group_id] = MessageBuffer()
            
            buffer = self._buffers[group_id]
            
            # 追加消息到缓冲池
            buffer.messages.append(msg)
            
            # 限制上下文消息数量，防止过长
            if len(buffer.messages) > self._max_context_messages:
                buffer.messages = buffer.messages[-self._max_context_messages:]
            
            logger.debug(f"[群{group_id}] 消息入池: {msg.raw_message[:30]}...")
            
            # 取消旧的定时器任务
            if buffer.timer_task and not buffer.timer_task.done():
                buffer.timer_task.cancel()
                logger.debug(f"[群{group_id}] 防抖定时器重置")
            
            # 【防抖重置】创建新的防抖定时器
            buffer.timer_task = asyncio.create_task(
                self._debounce_trigger(group_id, client)
            )
    
    async def _debounce_trigger(self, group_id: int, client: NapCatClient) -> None:
        """
        防抖触发器
        
        等待 debounce_seconds 秒后触发处理。
        如果在等待期间被取消（有新消息），则不执行处理。
        """
        try:
            # 【防抖等待】配置的秒数内无新消息才触发
            await asyncio.sleep(self._debounce_seconds)
            
            # 时间到，提取并处理缓冲池中的消息
            await self._process_buffer(group_id, client)
            
        except asyncio.CancelledError:
            # 被新消息取消，正常情况，不做任何处理
            logger.debug(f"[群{group_id}] 防抖被新消息重置")
            raise
    
    async def _process_buffer(self, group_id: int, client: NapCatClient) -> None:
        """
        处理缓冲池中的消息
        
        【双模型管线】
        1. 拼装上下文
        2. 调用导演模型生成状态
        3. 调用演员模型生成回复
        4. 物理层模拟发送
        """
        lock = self._get_lock(group_id)
        
        async with lock:
            buffer = self._buffers.get(group_id)
            if not buffer or not buffer.messages:
                return
            
            # 提取消息（消费掉缓冲池）
            messages = buffer.messages.copy()
            
            logger.info(f"[群{group_id}] 防抖结束，处理 {len(messages)} 条消息")

        force_reply = any(msg.is_at_me for msg in messages)
        # 【掷骰子逻辑】如果没有人 @ 我，才去乖乖摇骰子
        if not force_reply and random.random() > self._reply_probability:
            logger.info(f"[群{group_id}] 掷骰子失败 (概率 {self._reply_probability})，本次保持沉默，只听不说。")
            return
        elif force_reply:
            logger.info(f"[群{group_id}] 🔔 触发 @ 必回 ，无视概率，强制回复")
        
        # 1. 导演看全景图（完整的 50 条记忆）
        full_context = self._build_context(messages)
        
        # 2. 演员看纯净短切片（只看最近8条纯用户发言）
        actor_context = self._build_actor_context(messages, max_recent=8)

        # 🧠 【新增】：捞出机器人自己最近说过的 3 句话，用作防重复警告
        recent_self_msgs = [msg.raw_message for msg in messages if msg.sender_name == "我"][-3:]
        recent_self_str = "\n".join(recent_self_msgs) if recent_self_msgs else "无"
        
        try:
            # 【导演模型】推演核心逻辑和状态 (使用 full_context)
            logger.info(f"[群{group_id}] 调用导演模型 ({self._director_model})...")
            director_prompt = self._build_director_prompt(full_context)
            strategic_intent = await ask_llm(
                model_name=self._director_model,
                prompt_content=director_prompt
            )
            
            # 检查导演模型是否返回错误
            if strategic_intent.startswith("[LLM"):
                logger.error(f"[群{group_id}] 导演模型调用失败: {strategic_intent}")
                return
            
            logger.info(f"[群{group_id}] 导演指令: {strategic_intent}")
            
            # 【演员模型】根据指令生成最终回复 (使用 actor_context！)
            logger.info(f"[群{group_id}] 调用演员模型 ({self._actor_model})...")
            actor_prompt = self._build_actor_prompt(actor_context, strategic_intent, recent_self_str)
            reply = await ask_llm(
                model_name=self._actor_model,
                prompt_content=actor_prompt
            )
            
            # 检查演员模型是否返回错误
            if reply.startswith("[LLM"):
                logger.error(f"[群{group_id}] 演员模型调用失败: {reply}")
                return
            
            logger.info(f"[群{group_id}] 演员最终回复: {reply}")
            
            # 【物理层模拟】分段发送
            await self._send_with_typing_simulation(group_id, reply, client)

            # 🧠 【新增：记忆烙印】把自己说的话也记进历史池
            async with lock:
                # 重新获取 buffer，因为在 await 大模型期间，可能又有新消息进来了
                buffer = self._buffers.get(group_id)
                if buffer is not None:
                    bot_msg = GroupMessage(
                        group_id=group_id,
                        user_id=0,  # 随便给个 0 代表机器人自己
                        raw_message=reply.replace('|', ''),  # 把分段符去掉存入记忆
                        sender_name="我",  # 告诉大模型这是它自己说的话
                        timestamp=time.time()
                    )
                    buffer.messages.append(bot_msg)
                    
                    # 严格控制记忆容量，顶出最老的消息
                    if len(buffer.messages) > self._max_context_messages:
                        buffer.messages = buffer.messages[-self._max_context_messages:]
            
        except Exception as e:
            logger.error(f"[群{group_id}] 处理异常: {type(e).__name__}: {e}")
    
    def _build_context(self, messages: List[GroupMessage]) -> str:
        """
        构建上下文字符串
        
        将多条消息格式化为对话上下文。
        """
        lines = [msg.format_context() for msg in messages]
        return "\n".join(lines)
    
    def _build_actor_context(self, messages: List[GroupMessage], max_recent: int = 8) -> str:
        """
        【架构级隔离】构建演员专属的纯净上下文
        
        核心逻辑：
        1. 过滤掉机器人自己 (sender_name="我") 过去的所有发言，彻底切断自我洗脑的路径。
        2. 只保留最近的 max_recent 条真实用户的发言，让演员只针对眼前的对话做出反应。
        """
        # 过滤掉机器人自己的历史发言
        pure_user_messages = [msg for msg in messages if msg.sender_name != "我"]
        
        # 只取最近的几条
        recent_messages = pure_user_messages[-max_recent:] if pure_user_messages else []
        
        lines = [msg.format_context() for msg in recent_messages]
        return "\n".join(lines)
    
    def _build_director_prompt(self, context: str) -> str:
        """
        极简版导演提示词 (禁止写台词)
        """
        return f"""阅读以下群聊记录，用一句话完成两个任务：
1. 概括群友最新正在聊的具体话题。
2. 指示回复的【情绪和立场】。
（禁止写出具体的台词！禁止教演员怎么说话！只能给出方向）

【群聊记录】
{context}

直接输出这句话："""

    def _build_actor_prompt(self, context: str, strategic_intent: str, recent_self: str = "") -> str:
        """
        极简版演员提示词
        """
        return f"""你是这个QQ群里的普通群友。

【回复方向】
{strategic_intent}

【你最近说过的话】
{recent_self}
（不能重复你刚说过的意思和词汇！请换个角度接话！）

【最近的聊天记录】
{context}

【要求】
1. 参考回复方向，用符合情景的语气接一句话。
2. 绝对不要用括号写内心戏，不要有任何机器感。
3. 如果想分段发，用 | 符号分隔。

直接输出你的回复文字："""
    
    async def _send_with_typing_simulation(
        self,
        group_id: int,
        reply: str,
        client: NapCatClient
    ) -> None:
        """
        【物理层模拟】模拟真人打字发送
        
        1. 去除末尾句号
        2. 按 | 分段
        3. 每段之间随机延迟（配置的延迟范围）
        """
        # 去除末尾的句号
        reply = reply.rstrip('。')
        
        # 按 | 分段
        segments = [s.strip() for s in reply.split('|') if s.strip()]
        
        if not segments:
            return
        
        logger.info(f"[群{group_id}] 准备发送 {len(segments)} 段消息")
        
        for i, segment in enumerate(segments):
            # 发送消息
            try:
                await client.send_group_msg(group_id, segment)
                logger.info(f"[群{group_id}] 已发送 [{i+1}/{len(segments)}]: {segment}")
            except Exception as e:
                logger.error(f"[群{group_id}] 发送失败: {e}")
                continue
            
            # 如果不是最后一段，进行随机延迟（模拟打字）
            if i < len(segments) - 1:
                delay = random.uniform(self._typing_delay_min, self._typing_delay_max)
                logger.debug(f"[群{group_id}] 模拟打字中... ({delay:.2f}s)")
                await asyncio.sleep(delay)


# ============================================================
# 核心调度器
# ============================================================

class BotCore:
    """
    机器人核心调度器
    
    负责管理处理器的注册、消息分发、以及整体生命周期的控制。
    """
    
    def __init__(self, config: Optional[BotConfig] = None):
        """
        初始化核心调度器
        
        Args:
            config: 机器人配置对象，如果为 None 则从默认路径加载
        """
        self._config = config or BotConfig.from_file()
        self._napcat_client: Optional[NapCatClient] = None
        self._handler_chain: Optional[MessageHandler] = None
        self._running: bool = False
    
    @property
    def config(self) -> BotConfig:
        """获取当前配置"""
        return self._config
    
    def register_handler(self, handler: MessageHandler) -> None:
        """
        注册消息处理器
        
        【处理器注册机制】
        将处理器添加到责任链的末端。
        如果是第一个处理器，设置为链头；
        否则追加到现有链的末尾。
        
        Args:
            handler: 要注册的处理器实例
        """
        if self._handler_chain is None:
            self._handler_chain = handler
            logger.info(f"处理器注册: {handler.__class__.__name__} (链头)")
        else:
            # 遍历到链尾并追加
            current = self._handler_chain
            while current._next_handler is not None:
                current = current._next_handler
            current.set_next(handler)
            logger.info(f"处理器注册: {handler.__class__.__name__} (追加到链尾)")
    
    async def _on_message(self, event: dict) -> None:
        """
        消息回调入口
        
        当 NapCat 收到消息时调用此方法，
        将消息分发给处理器链处理。
        
        Args:
            event: 原始事件数据
        """
        # 忽略非消息事件（如心跳等）
        post_type = event.get('post_type')
        if post_type != 'message':
            return
        
        # 分发给处理器链
        if self._handler_chain:
            try:
                await self._handler_chain.process(event, self._napcat_client)
            except Exception as e:
                logger.error(f"处理器链执行异常: {type(e).__name__}: {e}")
    
    async def start(self) -> None:
        """
        启动机器人
        
        初始化 NapCat 客户端并开始监听消息。
        """
        if self._running:
            logger.warning("机器人已在运行中")
            return
        
        self._running = True
        
        # 获取 NapCat 客户端实例
        self._napcat_client = get_client()
        
        logger.info("=" * 50)
        logger.info(f"{self._config.name} v{self._config.version} 启动")
        logger.info("=" * 50)
        
        # 显示处理器链信息
        if self._handler_chain:
            handlers = []
            current = self._handler_chain
            while current:
                handlers.append(current.__class__.__name__)
                current = current._next_handler
            logger.info(f"处理器链: {' -> '.join(handlers)}")
        else:
            logger.warning("未注册任何消息处理器！")
        
        try:
            # 启动 NapCat 客户端（阻塞式）
            await self._napcat_client.start(self._on_message)
        except asyncio.CancelledError:
            logger.info("主任务被取消，准备退出...")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """
        停止机器人
        
        关闭 NapCat 客户端。
        """
        if not self._running:
            return
        
        logger.info(f"正在停止 {self._config.name}...")
        self._running = False
        
        if self._napcat_client:
            await self._napcat_client.stop()
        
        logger.info(f"{self._config.name} 已停止")


# ============================================================
# 全局实例与便捷函数
# ============================================================

_bot_core: Optional[BotCore] = None


def get_bot(config: Optional[BotConfig] = None) -> BotCore:
    """
    获取全局 BotCore 实例
    
    Args:
        config: 可选的配置对象，仅首次调用时有效
        
    Returns:
        BotCore 实例
    """
    global _bot_core
    if _bot_core is None:
        _bot_core = BotCore(config)
    return _bot_core


# ============================================================
# 主入口
# ============================================================

async def main():
    """
    主入口函数
    
    【启动流程】
    1. 加载配置文件
    2. 配置日志系统
    3. 创建 BotCore 实例
    4. 注册处理器（按优先级顺序，配置注入）
    5. 启动机器人
    """
    # 【配置加载】从 main_config.json 加载配置
    config = BotConfig.from_file()
    
    # 配置日志系统
    config.setup_logging()
    
    logger.info(f"配置加载完成: {config.name} v{config.version}")
    logger.info(
        f"行为参数: 防抖={config.debounce_seconds}s, "
        f"打字延迟={config.typing_delay_min}~{config.typing_delay_max}s"
    )
    logger.info(
        f"模型配置: 导演={config.director_model}, 演员={config.actor_model}"
    )
    
    # 获取核心调度器实例（传入配置）
    bot = get_bot(config)
    
    # 【处理器注册】使用配置工厂方法创建处理器
    # 所有参数从配置文件注入，不再硬编码
    group_llm_handler = GroupLLMHandler.from_config(config)
    bot.register_handler(group_llm_handler)
    
    # 后续可在此处添加更多处理器：
    # bot.register_handler(CommandHandler())
    # bot.register_handler(ImageHandler())
    
    # 启动机器人
    await bot.start()


def run():
    """
    运行入口
    
    处理优雅启动与退出逻辑。
    """
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("\n收到键盘中断信号 (Ctrl+C)")
    except Exception as e:
        logger.error(f"运行时异常: {type(e).__name__}: {e}")
    finally:
        # 清理资源：取消所有任务
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        
        # 等待所有任务完成取消
        if tasks:
            loop.run_until_complete(
                asyncio.gather(*tasks, return_exceptions=True)
            )
        
        loop.close()
        logger.info("事件循环已关闭，程序退出")


if __name__ == "__main__":
    run()
