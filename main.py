"""
LessBot 核心调度中枢 (Core Dispatcher)

该模块是整个机器人的大脑，负责：
1. 消息分发与处理器路由（责任链模式）
2. 群聊消息的防抖缓冲（滑动窗口）
3. 双模型管线调用（导演-演员架构）
4. 物理层模拟（真人化回复节奏)
5. 配置驱动的外部化参数管理
"""

import asyncio
import logging
from typing import Optional

from napcat_client import get_client, NapCatClient
from config import BotConfig
from handlers import MessageHandler, GroupLLMHandler

# ============================================================
# 日志配置
# ============================================================

logger = logging.getLogger("LessBot")


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
        否则遍历到链尾并追加。
        
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
