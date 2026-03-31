"""
LessBot 配置管理模块

封装所有可配置的行为参数，支持从 JSON 文件加载。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
