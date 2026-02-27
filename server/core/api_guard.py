"""
LLM Guard Provider 抽象与实现
"""

import os
import re
import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from io import BytesIO

import cv2
import numpy as np


DecisionType = Literal["approve", "deny", "uncertain"]


@dataclass
class LLMDecision:
    """LLM Guard 决策结果"""
    decision: DecisionType
    confidence: Optional[float] = None
    reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None


def resolve_env_vars(value: str) -> str:
    """
    解析环境变量占位符 ${VAR_NAME}

    Args:
        value: 配置值（可能包含 ${VAR_NAME}）

    Returns:
        解析后的值
    """
    if not isinstance(value, str):
        return value

    pattern = re.compile(r'\$\{([^}]+)\}')

    def replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, "")

    return pattern.sub(replacer, value)


def encode_image(image: np.ndarray, max_size: int = 1024) -> str:
    """
    将 numpy 图像编码为 base64

    Args:
        image: BGR numpy array
        max_size: 最大边长（压缩以节省token）

    Returns:
        base64 编码的 JPEG 图像
    """
    # 按比例缩放到 max_size
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    # 编码为 JPEG
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def parse_llm_response(text: str, mode: str = "targeted") -> Dict[str, Any]:
    """
    解析 LLM 响应，提取决策和理由

    Args:
        text: LLM 响应文本
        mode: "targeted" (目标验证) 或 "global" (全局验证)

    Targeted 支持格式：
    - "RISK_CONFIRMED\n理由" → approve
    - "FALSE_ALARM\n理由" → deny
    - "UNCERTAIN\n理由" → uncertain
    - 也兼容旧格式 approve/deny/uncertain

    Global 支持格式：
    - "RISK_FOUND\n问题描述" → approve
    - "SAFE\n说明" → deny
    - "UNCERTAIN\n理由" → uncertain

    Returns:
        {"decision": "approve/deny/uncertain", "reason": "理由"}
    """
    text = text.strip()
    text_lower = text.lower()

    if mode == "global":
        # 全局验证模式
        keyword_map = {
            "risk_found": "approve",  # 发现风险 → approve（触发告警）
            "safe": "deny",  # 画面安全 → deny（不触发）
            "uncertain": "uncertain",
        }
    else:
        # 目标验证模式（默认）
        keyword_map = {
            "risk_confirmed": "approve",
            "false_alarm": "deny",
            "uncertain": "uncertain",
        }

    for keyword, decision in keyword_map.items():
        if text_lower.startswith(keyword):
            remaining = text[len(keyword):].strip()
            if remaining.startswith(":"):
                remaining = remaining[1:].strip()
            elif remaining.startswith("\n"):
                remaining = remaining[1:].strip()
            return {"decision": decision, "reason": remaining or None}

    # 兼容旧格式：approve / deny / uncertain
    for decision in ["approve", "deny", "uncertain"]:
        if text_lower.startswith(decision):
            remaining = text[len(decision):].strip()
            if remaining.startswith(":"):
                remaining = remaining[1:].strip()
            return {"decision": decision, "reason": remaining or None}

    # 兜底：返回 uncertain
    return {"decision": "uncertain", "reason": text[:100] if text else None}


def draw_target_box(
    image: np.ndarray,
    target_bbox: List[float],
    color: tuple = (0, 0, 255),
    thickness: int = 3
) -> np.ndarray:
    """在图像上绘制目标框（与DangerJudge相同逻辑）"""
    img_with_box = image.copy()
    x1, y1, x2, y2 = map(int, target_bbox)
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), color, thickness)
    return img_with_box


def build_vlm_prompt(
    expert_config: dict,
    temporal_info: Optional[Dict[str, Any]] = None,
    zone_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    构建VLM风格的结构化Prompt（与DangerJudge._build_prompt相同逻辑）
    用于云端API替代本地VLM时使用
    """
    scene_ids = expert_config.get("scene_ids", [])
    scene_desc = ", ".join(scene_ids) if scene_ids else "未知场景"

    prompt_parts = []

    # Part 1: 场景说明
    prompt_parts.append(f"场景：{scene_desc}")
    prompt_parts.append("目标：画面中红框标记的对象")
    prompt_parts.append("")

    # Part 2: 检测结果（时序信息）
    if temporal_info:
        prompt_parts.append("检测结果：")
        if "attribute_descriptions" in temporal_info:
            for desc in temporal_info["attribute_descriptions"]:
                prompt_parts.append(f"  - {desc}")
        if "duration" in temporal_info:
            prompt_parts.append(f"  - 已出现 {temporal_info['duration']} 秒")
        prompt_parts.append("")

    # Part 3: 区域信息
    if zone_info and zone_info.get("zone_configured"):
        zone_name = zone_info.get("zone_name", "unknown")
        zone_type = zone_info.get("zone_type", "undefined")
        zone_type_labels = {
            "danger_zone": "危险作业区",
            "safe_zone": "安全区/休息区",
            "rest_zone": "休息区",
            "restricted_zone": "禁入区域",
            "undefined": "未定义区域",
            "unknown": "未知区域"
        }
        zone_label = zone_type_labels.get(zone_type, zone_type)
        prompt_parts.append(f"当前位置：{zone_name}（{zone_label}）")
        prompt_parts.append("")

    # Part 4: 判断请求（使用专家配置的模板）
    custom_template = expert_config.get("vlm_prompt_template", "")
    if custom_template:
        prompt_parts.append("判断标准：")
        prompt_parts.append(custom_template)
        prompt_parts.append("")

    # Part 5: 输出格式要求
    prompt_parts.append("请综合以上信息判断：这是否为真正需要告警的危险行为？")
    prompt_parts.append("")
    prompt_parts.append("输出格式：")
    prompt_parts.append("第一行：DANGER 或 SAFE")
    prompt_parts.append("第二行：判断理由（说明为什么是/不是危险）")

    return "\n".join(prompt_parts)


def parse_vlm_response(response_text: str) -> Dict[str, Any]:
    """
    解析VLM风格的响应（DANGER/SAFE/UNCERTAIN格式）
    返回与parse_llm_response兼容的格式
    """
    lines = response_text.strip().split('\n', 1)
    first_line = lines[0].strip().upper()

    if "DANGER" in first_line:
        decision = "approve"  # DANGER → approve（触发告警）
    elif "SAFE" in first_line:
        decision = "deny"     # SAFE → deny（不触发）
    else:
        decision = "uncertain"

    reason = lines[1].strip() if len(lines) > 1 else None
    return {"decision": decision, "reason": reason}


class LLMProvider:
    """LLM Guard 统一接口"""

    name = "base"

    def __init__(self, config: dict):
        self.config = config
        self.timeout_s = config.get("timeout_s", 10)
        self.retry = config.get("retry", 2)
        self.threshold = config.get("threshold")

        # 获取公共默认 prompts
        default_prompts = config.get("default_prompts", {})

        # 获取 provider 特定配置
        provider_config = config.get(self.name, {})
        self.api_key = resolve_env_vars(provider_config.get("api_key", ""))
        self.base_url = resolve_env_vars(provider_config.get("base_url", ""))
        self.model = provider_config.get("model", "")
        self.max_tokens = provider_config.get("max_tokens", 200)
        self.temperature = provider_config.get("temperature", 0.0)

        # Prompt 继承逻辑：优先使用 provider 配置，否则使用 default_prompts
        self.system_prompt = provider_config.get(
            "system_prompt",
            default_prompts.get("system_prompt", "")
        )
        self.user_prompt_template = provider_config.get(
            "user_prompt_template",
            default_prompts.get("user_prompt_template", "")
        )

    def infer(self, image: np.ndarray, context: Dict[str, Any]) -> LLMDecision:
        """
        推理接口（需要子类实现）

        Args:
            image: BGR numpy array
            context: 上下文信息（expert_id, risk_type, fused_conf等）

        Returns:
            LLMDecision
        """
        raise NotImplementedError

    def _build_decision(
        self,
        confidence: Optional[float] = None,
        decision: Optional[DecisionType] = None,
        reason: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None
    ) -> LLMDecision:
        """构建决策结果"""
        if decision is None:
            if confidence is not None and self.threshold is not None:
                decision = "approve" if confidence >= self.threshold else "deny"
            else:
                decision = "uncertain"
        elif decision not in ("approve", "deny", "uncertain"):
            raise ValueError(f"Unsupported decision: {decision}")
        return LLMDecision(
            decision=decision,
            confidence=confidence,
            reason=reason,
            raw=raw,
            provider=self.name
        )

    def _retry_call(self, func, *args, **kwargs):
        """重试机制"""
        last_error = None
        for attempt in range(self.retry + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.retry:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
        raise last_error

    def infer_as_vlm(
        self,
        image: np.ndarray,
        expert_config: dict,
        track_id: int,
        target_bbox: List[float],
        temporal_info: Optional[Dict[str, Any]] = None,
        zone_info: Optional[Dict[str, Any]] = None
    ) -> LLMDecision:
        """
        作为本地VLM替代时的推理接口
        使用与DangerJudge相同的Prompt构建逻辑

        Args:
            image: BGR numpy array
            expert_config: 专家配置（包含vlm_prompt_template）
            track_id: 跟踪ID
            target_bbox: 目标边界框 [x1, y1, x2, y2]
            temporal_info: 时序信息（持续时间、状态变化等）
            zone_info: 区域信息（zone_name, zone_type等）

        Returns:
            LLMDecision（decision映射：approve=DANGER, deny=SAFE, uncertain=UNCERTAIN）
        """
        # 1. 绘制红框标记目标
        img_with_box = draw_target_box(image, target_bbox)

        # 2. 构建VLM风格的结构化Prompt
        vlm_prompt = build_vlm_prompt(expert_config, temporal_info, zone_info)

        # 3. 创建特殊context，使用VLM风格的prompt
        context = {"_vlm_prompt": vlm_prompt}

        # 4. 调用推理（子类实现）
        return self.infer(img_with_box, context)

    def _format_user_prompt(self, context: Dict[str, Any]) -> str:
        """格式化用户提示词"""
        # 支持VLM风格的prompt直接传递
        if "_vlm_prompt" in context:
            return context["_vlm_prompt"]
        return self.user_prompt_template.format(**context)


class ClaudeProvider(LLMProvider):
    """Claude LLM Provider（Anthropic API）"""

    name = "claude"

    def __init__(self, config: dict):
        super().__init__(config)
        if not self.base_url:
            self.base_url = "https://api.anthropic.com"

    def infer(self, image: np.ndarray, context: Dict[str, Any]) -> LLMDecision:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "需要安装 anthropic SDK: pip install anthropic"
            )

        if not self.api_key:
            return self._build_decision(
                decision="uncertain",
                reason="ANTHROPIC_API_KEY 未配置"
            )

        # 编码图像
        image_b64 = encode_image(image)

        # 构建消息
        user_prompt = self._format_user_prompt(context)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        # 调用 API
        def _call_api():
            client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages,
                timeout=self.timeout_s
            )
            return response

        try:
            response = self._retry_call(_call_api)
            response_text = response.content[0].text
            parsed = parse_llm_response(response_text)

            return self._build_decision(
                decision=parsed["decision"],
                reason=parsed["reason"],
                raw={"response": response_text}
            )
        except Exception as e:
            return self._build_decision(
                decision="uncertain",
                reason=f"API调用失败: {str(e)[:50]}"
            )


class QwenVLProvider(LLMProvider):
    """Qwen-VL Provider（阿里云通义千问）"""

    name = "qwen_vl"

    def __init__(self, config: dict):
        super().__init__(config)
        if not self.base_url:
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def infer(self, image: np.ndarray, context: Dict[str, Any]) -> LLMDecision:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "需要安装 openai SDK: pip install openai"
            )

        if not self.api_key:
            return self._build_decision(
                decision="uncertain",
                reason="QWEN_API_KEY 未配置"
            )

        # 编码图像
        image_b64 = encode_image(image)

        # 构建消息（OpenAI 兼容格式）
        user_prompt = self._format_user_prompt(context)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        # 调用 API
        def _call_api():
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response

        try:
            response = self._retry_call(_call_api)
            response_text = response.choices[0].message.content
            parsed = parse_llm_response(response_text)

            return self._build_decision(
                decision=parsed["decision"],
                reason=parsed["reason"],
                raw={"response": response_text}
            )
        except Exception as e:
            return self._build_decision(
                decision="uncertain",
                reason=f"API调用失败: {str(e)[:50]}"
            )


class OpenAIProvider(LLMProvider):
    """OpenAI Provider（GPT-4o/GPT-4-turbo）"""

    name = "openai"

    def __init__(self, config: dict):
        super().__init__(config)
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

    def infer(self, image: np.ndarray, context: Dict[str, Any]) -> LLMDecision:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "需要安装 openai SDK: pip install openai"
            )

        if not self.api_key:
            return self._build_decision(
                decision="uncertain",
                reason="OPENAI_API_KEY 未配置"
            )

        # 编码图像
        image_b64 = encode_image(image)

        # 构建消息
        user_prompt = self._format_user_prompt(context)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        # 调用 API
        def _call_api():
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response

        try:
            response = self._retry_call(_call_api)
            response_text = response.choices[0].message.content
            parsed = parse_llm_response(response_text)

            return self._build_decision(
                decision=parsed["decision"],
                reason=parsed["reason"],
                raw={"response": response_text}
            )
        except Exception as e:
            return self._build_decision(
                decision="uncertain",
                reason=f"API调用失败: {str(e)[:50]}"
            )


def build_provider(config: dict) -> LLMProvider:
    """根据配置实例化 Provider"""
    import yaml
    from pathlib import Path

    # 加载 API 密钥配置
    api_config_path = config.get("api_config_path", "llm_api.yaml")

    # 解析路径（相对于 llm_config.yaml 所在目录）
    if not Path(api_config_path).is_absolute():
        # 假设 config 是从 server/llm_config.yaml 加载的
        config_dir = Path(__file__).parent.parent  # server/
        api_config_path = config_dir / api_config_path

    # 加载 API 配置
    if Path(api_config_path).exists():
        with open(api_config_path, 'r', encoding='utf-8') as f:
            api_config = yaml.safe_load(f)

        # 将 API 密钥注入到对应 provider 配置
        for provider_name in ["claude", "qwen_vl", "openai"]:
            if provider_name in api_config:
                if provider_name not in config:
                    config[provider_name] = {}
                config[provider_name]["api_key"] = api_config[provider_name].get("api_key", "")
                config[provider_name]["base_url"] = api_config[provider_name].get("base_url", "")

    # 实例化 Provider
    provider_name = config.get("provider")
    providers = {
        ClaudeProvider.name: ClaudeProvider,
        QwenVLProvider.name: QwenVLProvider,
        OpenAIProvider.name: OpenAIProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return providers[provider_name](config)

