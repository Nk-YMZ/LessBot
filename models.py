"""
LessBot 数据模型定义

定义群消息数据结构和消息缓冲池。
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional


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
