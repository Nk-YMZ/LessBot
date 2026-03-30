"""
大模型调用中心模块 (LLM Caller Module)

该模块提供了统一的异步接口，用于调用不同配置的大模型服务。
支持多种 API 格式（OpenAI 兼容、Anthropic 兼容等），通过配置文件灵活管理。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx


class LLMCallerError(Exception):
    """大模型调用相关的自定义异常基类"""
    pass


class ModelNotFoundError(LLMCallerError):
    """请求的模型在配置中不存在"""
    pass


class LLMResponseError(LLMCallerError):
    """模型响应解析失败"""
    pass


class LLMCaller:
    """
    大模型调用中心类
    
    负责加载配置、管理 HTTP 客户端、处理不同提供商的 API 调用格式。
    采用单例模式管理配置，避免重复加载。
    """
    
    _instance: Optional['LLMCaller'] = None
    _config: Optional[dict] = None
    _config_path: Optional[Path] = None
    
    def __new__(cls) -> 'LLMCaller':
        """单例模式：确保全局只有一个配置实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """初始化时并不立即加载配置，采用懒加载策略"""
        if self._config is not None:
            return
    
    def _resolve_env_vars(self, value: str) -> str:
        """
        解析字符串中的环境变量占位符
        
        支持格式：${ENV_VAR_NAME} 或 ${ENV_VAR_NAME:default_value}
        
        Args:
            value: 可能包含环境变量占位符的字符串
            
        Returns:
            解析后的字符串
        """
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) or ''
            return os.environ.get(var_name, default_value)
        
        return re.sub(pattern, replace_env_var, value)
    
    def _resolve_internal_refs(self, config: dict, value: str) -> str:
        """
        解析配置内部的引用，如 ${api_keys.openai}
        
        Args:
            config: 完整配置字典
            value: 可能包含内部引用的字符串
            
        Returns:
            解析后的字符串
        """
        # 匹配 ${api_keys.xxx} 或 ${global_settings.xxx} 格式
        pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\}'
        
        def replace_ref(match: re.Match) -> str:
            section = match.group(1)
            key = match.group(2)
            if section in config and key in config[section]:
                return str(config[section][key])
            return match.group(0)  # 保持原样
        
        return re.sub(pattern, replace_ref, value)
    
    def _resolve_config_values(self, config: dict, raw_config: dict) -> Any:
        """
        递归解析配置中所有值（先解析内部引用，再解析环境变量）
        
        Args:
            config: 已解析的配置字典（用于内部引用）
            raw_config: 原始配置对象
            
        Returns:
            解析后的配置对象
        """
        if isinstance(raw_config, dict):
            return {k: self._resolve_config_values(config, v) for k, v in raw_config.items()}
        elif isinstance(raw_config, list):
            return [self._resolve_config_values(config, item) for item in raw_config]
        elif isinstance(raw_config, str):
            # 先解析内部引用，再解析环境变量
            result = self._resolve_internal_refs(config, raw_config)
            result = self._resolve_env_vars(result)
            return result
        else:
            return raw_config
    
    def load_config(self, config_path: Optional[str] = None) -> dict:
        """
        加载并解析模型配置文件
        
        配置文件优先级：
        1. 显式指定的路径
        2. 环境变量 LLM_CONFIG_PATH
        3. 默认路径 ./models_config.json
        
        Args:
            config_path: 配置文件路径（可选）
            
        Returns:
            解析后的配置字典
        """
        # 确定配置文件路径
        if config_path:
            path = Path(config_path)
        elif os.environ.get('LLM_CONFIG_PATH'):
            path = Path(os.environ['LLM_CONFIG_PATH'])
        else:
            path = Path(__file__).parent / 'models_config.json'
        
        self._config_path = path
        
        # 读取并解析 JSON
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)
        
        # 两阶段解析：先构建基础结构，再解析引用和环境变量
        self._config = raw_config
        self._config = self._resolve_config_values(raw_config, raw_config)
        
        return self._config
    
    def _ensure_config_loaded(self) -> dict:
        """确保配置已加载，如果未加载则使用默认路径"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_model_config(self, model_name: str) -> dict:
        """
        获取指定模型的配置信息
        
        Args:
            model_name: 模型名称标识符
            
        Returns:
            该模型的完整配置字典
            
        Raises:
            ModelNotFoundError: 模型不存在于配置中
        """
        config = self._ensure_config_loaded()
        
        models = config.get('models', {})
        if model_name not in models:
            available = list(models.keys())
            raise ModelNotFoundError(
                f"模型 '{model_name}' 未在配置中找到。可用模型：{available}"
            )
        
        return models[model_name]
    
    def _build_headers(self, model_config: dict) -> dict:
        """
        根据模型配置构建请求头
        
        Args:
            model_config: 单个模型的配置字典
            
        Returns:
            格式化后的请求头字典
        """
        provider = model_config.get('provider', 'openai')
        api_key = model_config.get('api_key', '')
        
        if provider == 'anthropic':
            return {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2024-01-01'
            }
        else:
            # 默认 OpenAI 兼容格式
            return {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
    
    def _build_request_body(
        self, 
        model_config: dict, 
        prompt_content: str,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        构建请求体（根据 provider 类型）
        
        Args:
            model_config: 模型配置
            prompt_content: 用户输入的提示内容
            system_prompt: 可选的系统提示词
            
        Returns:
            请求体字典
        """
        provider = model_config.get('provider', 'openai')
        model_name = model_config.get('model', 'gpt-3.5-turbo')
        
        if provider == 'anthropic':
            body = {
                'model': model_name,
                'messages': [{'role': 'user', 'content': prompt_content}],
                'max_tokens': 4096
            }
            if system_prompt:
                body['system'] = system_prompt
        else:
            # OpenAI 兼容格式
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt_content})
            
            body = {
                'model': model_name,
                'messages': messages
            }
        
        return body
    
    def _parse_response(self, response_data: dict, provider: str) -> str:
        """
        解析 API 响应
        
        Args:
            response_data: API 返回的 JSON 数据
            provider: 提供商类型
            
        Returns:
            提取出的文本内容
        """
        try:
            if provider == 'anthropic':
                # Anthropic 响应格式
                content_list = response_data.get('content', [])
                if not content_list:
                    raise LLMResponseError("响应中没有 content 字段或为空")
                
                text_parts = []
                for item in content_list:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                
                content = ''.join(text_parts)
            else:
                # OpenAI 兼容格式
                choices = response_data.get('choices', [])
                if not choices:
                    raise LLMResponseError("响应中没有 choices 字段或为空")
                
                message = choices[0].get('message', {})
                content = message.get('content', '')
            
            if not content:
                raise LLMResponseError("响应内容为空")
            
            return content
            
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseError(f"解析响应失败: {e}")
    
    async def _make_request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        body: dict,
        timeout: float,
        max_retries: int,
        retry_delay: float
    ) -> dict:
        """
        带重试机制的异步 HTTP 请求
        
        Args:
            client: httpx 异步客户端
            url: 请求 URL
            headers: 请求头
            body: 请求体
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            
        Returns:
            解析后的 JSON 响应
        """
        import asyncio
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 429:
                    last_error = LLMCallerError(
                        f"API 速率限制 (429)，尝试 {attempt + 1}/{max_retries + 1}"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                        
                elif response.status_code >= 500:
                    last_error = LLMCallerError(
                        f"服务器错误 ({response.status_code})，"
                        f"尝试 {attempt + 1}/{max_retries + 1}"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        continue
                        
                elif response.status_code >= 400:
                    error_detail = response.text[:200]
                    raise LLMCallerError(
                        f"API 请求失败 ({response.status_code}): {error_detail}"
                    )
                
                raise LLMCallerError(f"意外的 HTTP 状态码: {response.status_code}")
                
            except httpx.TimeoutException:
                last_error = LLMCallerError(
                    f"请求超时 ({timeout}s)，尝试 {attempt + 1}/{max_retries + 1}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                    
            except httpx.NetworkError as e:
                last_error = LLMCallerError(
                    f"网络错误: {e}，尝试 {attempt + 1}/{max_retries + 1}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                    
            except json.JSONDecodeError as e:
                raise LLMCallerError(f"响应 JSON 解析失败: {e}")
        
        raise last_error or LLMCallerError("未知错误")


# ============================================================
# 模块级单例实例和统一接口
# ============================================================

_caller_instance: Optional[LLMCaller] = None


def _get_caller() -> LLMCaller:
    """获取全局 LLMCaller 单例实例"""
    global _caller_instance
    if _caller_instance is None:
        _caller_instance = LLMCaller()
    return _caller_instance


async def ask_llm(
    model_name: str, 
    prompt_content: str,
    system_prompt: Optional[str] = None,
    config_path: Optional[str] = None
) -> str:
    """
    统一的大模型调用接口（异步函数）
    
    这是主控模块应该调用的主要入口函数。
    
    Args:
        model_name: 模型标识符，对应 models_config.json 中的 key
        prompt_content: 发送给模型的提示内容
        system_prompt: 可选的系统提示词
        config_path: 可选的配置文件路径
        
    Returns:
        模型生成的文本回复
        
    Example:
        >>> response = await ask_llm(
        ...     model_name="deepseek-chat",
        ...     prompt_content="请用一句话解释量子计算"
        ... )
        >>> print(response)
    """
    caller = _get_caller()
    
    if config_path:
        caller.load_config(config_path)
    
    # 获取模型配置
    try:
        model_config = caller.get_model_config(model_name)
    except ModelNotFoundError:
        return f"[LLM 错误] 模型 '{model_name}' 未在配置中找到"
    
    # 获取全局设置
    full_config = caller._ensure_config_loaded()
    global_settings = full_config.get('global_settings', {})
    
    # 确定参数
    timeout = global_settings.get('default_timeout', 30)
    max_retries = global_settings.get('max_retries', 2)
    retry_delay = global_settings.get('retry_delay', 1.0)
    
    # 获取 API 信息
    api_url = model_config.get('api_url', '')
    provider = model_config.get('provider', 'openai')
    
    # 构建请求
    headers = caller._build_headers(model_config)
    body = caller._build_request_body(model_config, prompt_content, system_prompt)
    
    # 发送请求
    async with httpx.AsyncClient() as client:
        try:
            response_data = await caller._make_request_with_retry(
                client=client,
                url=api_url,
                headers=headers,
                body=body,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            return caller._parse_response(response_data, provider)
            
        except LLMCallerError as e:
            return f"[LLM 调用失败] {str(e)}"
            
        except Exception as e:
            return f"[LLM 未知错误] {type(e).__name__}: {str(e)}"


def reload_config(config_path: Optional[str] = None) -> None:
    """
    重新加载配置文件
    
    Args:
        config_path: 可选的新配置文件路径
    """
    global _caller_instance
    if _caller_instance is not None:
        _caller_instance._config = None
        _caller_instance.load_config(config_path)


def list_available_models() -> list[str]:
    """
    列出配置中所有可用的模型名称
    
    Returns:
        模型名称列表
    """
    caller = _get_caller()
    config = caller._ensure_config_loaded()
    return list(config.get('models', {}).keys())


# ============================================================
# 测试代码
# ============================================================

async def _test_llm_caller():
    """测试函数"""
    print("=" * 50)
    print("LLM Caller 模块测试")
    print("=" * 50)
    
    models = list_available_models()
    print(f"\n可用模型: {models}")
    
    print("\n测试调用 deepseek-chat 模型...")
    response = await ask_llm(
        model_name="deepseek-chat",
        prompt_content="Hello!"
    )
    print(f"响应: {response}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_llm_caller())
