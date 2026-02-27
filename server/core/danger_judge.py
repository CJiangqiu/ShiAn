"""
第5层：本地VLM危险判断
使用本地Qwen2-VL模型进行危险识别（Prompt驱动）
支持异步推理，不阻塞主流程
"""

import json
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional, List, Literal, Tuple

import cv2
import numpy as np

# 尝试导入VLM依赖（如果没有安装则优雅降级）
try:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    VLM_AVAILABLE = True
except ImportError as e:
    VLM_AVAILABLE = False
    VLM_IMPORT_ERROR = str(e)


JudgmentType = Literal["DANGER", "SAFE", "UNCERTAIN"]

# 异步任务的唯一标识 (expert_id, obj_id, rule_name)
VLMTaskKey = Tuple[str, int, str]


@dataclass
class DangerJudgment:
    """危险判断结果"""
    judgment: JudgmentType
    confidence: float
    reason: str
    track_id: int
    raw_response: Optional[str] = None


@dataclass
class AsyncVLMTask:
    """异步VLM任务"""
    key: VLMTaskKey
    future: Future
    submit_frame_id: int
    # 保存提交时的上下文，用于告警触发
    context: Optional[Dict[str, Any]] = None


class DangerJudge:
    """本地VLM危险判断器（支持异步推理）"""

    def __init__(
        self,
        models_dir: Path,
        device: str = "cuda",
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_workers: int = 1
    ):
        """
        初始化危险判断器

        Args:
            models_dir: 模型目录
            device: 设备 (cuda/cpu)
            model_name: VLM模型名称
            max_workers: 异步线程池大小（默认1，因为GPU推理通常不能并行）
        """
        self.models_dir = Path(models_dir)
        self.device = device
        self.model_name = model_name

        # 缓存专家配置
        self.expert_configs: Dict[str, dict] = {}

        # 异步推理支持
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_tasks: Dict[VLMTaskKey, AsyncVLMTask] = {}
        self._lock = Lock()

        # 初始化VLM模型
        self._init_vlm()

    def _init_vlm(self):
        """初始化本地VLM模型"""
        # 检查VLM依赖是否可用
        if not VLM_AVAILABLE:
            print(f"[DangerJudge] 警告: VLM依赖未安装")
            print(f"[DangerJudge] 错误: {VLM_IMPORT_ERROR}")
            print(f"[DangerJudge] 请安装: pip install transformers qwen-vl-utils torch")
            self.model = None
            self.processor = None
            return

        print(f"[DangerJudge] 正在加载本地VLM模型: {self.model_name}")

        # 查找本地模型
        load_path = self._find_local_model()
        if not load_path:
            print(f"[DangerJudge] 错误: 未找到本地VLM模型")
            print(f"[DangerJudge] 请运行: python train/scripts/5_build_model.py")
            self.model = None
            self.processor = None
            return

        try:
            print(f"[DangerJudge] 从本地加载: {load_path}")

            # 使用 AutoModelForImageTextToText 自动加载正确的模型类
            self.model = AutoModelForImageTextToText.from_pretrained(
                load_path,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            self.processor = AutoProcessor.from_pretrained(load_path)

            print(f"[DangerJudge] VLM模型加载完成")

        except Exception as e:
            print(f"[DangerJudge] 警告: VLM模型加载失败: {e}")
            self.model = None
            self.processor = None

    def _find_local_model(self) -> Optional[str]:
        """在本地查找VLM模型"""
        model_path = self.models_dir / self.model_name
        if model_path.exists() and (model_path / "config.json").exists():
            return str(model_path)
        return None

    def _load_expert_config(self, expert_id: str) -> Optional[dict]:
        """加载专家配置"""
        if expert_id in self.expert_configs:
            return self.expert_configs[expert_id]

        config_path = self.models_dir / expert_id / "expert_info.json"
        if not config_path.exists():
            print(f"[DangerJudge] 警告: 专家配置不存在: {config_path}")
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.expert_configs[expert_id] = config
                return config
        except Exception as e:
            print(f"[DangerJudge] 警告: 加载专家配置失败: {e}")
            return None

    def _draw_target_box(
        self,
        image: np.ndarray,
        target_bbox: List[float],
        color: tuple = (0, 0, 255),  # 红色
        thickness: int = 3
    ) -> np.ndarray:
        """在图像上绘制目标框"""
        img_with_box = image.copy()
        x1, y1, x2, y2 = map(int, target_bbox)
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), color, thickness)
        return img_with_box

    def _build_prompt(
        self,
        expert_config: dict,
        temporal_info: Optional[Dict[str, Any]] = None,
        zone_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建VLM提示词（结构化输入）

        Args:
            expert_config: 专家配置（包含vlm_prompt_template）
            temporal_info: 时序信息（持续时间、状态变化等）
            zone_info: 区域信息（zone_name, zone_type等）

        Returns:
            完整的提示词
        """
        # 获取场景描述
        scene_ids = expert_config.get("scene_ids", [])
        scene_desc = ", ".join(scene_ids) if scene_ids else "未知场景"

        # 构建结构化的 Prompt
        prompt_parts = []

        # Part 1: 场景说明
        prompt_parts.append(f"场景：{scene_desc}")
        prompt_parts.append("目标：画面中红框标记的对象")
        prompt_parts.append("")

        # Part 2: 检测结果（时序信息）
        if temporal_info:
            prompt_parts.append("检测结果：")

            # 人类可读的属性描述
            if "attribute_descriptions" in temporal_info:
                for desc in temporal_info["attribute_descriptions"]:
                    prompt_parts.append(f"  - {desc}")

            # 出现持续时间
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

        # Part 4: 判断请求（使用专家配置的模板，或使用默认模板）
        custom_template = expert_config.get("vlm_prompt_template", "")

        if custom_template:
            # 如果有自定义模板，使用它（但只用作判断标准，不再让VLM猜测事实）
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

    def _parse_vlm_response(self, response_text: str) -> tuple[JudgmentType, str]:
        """
        解析VLM响应

        预期格式：
        第一行：DANGER 或 SAFE
        第二行：理由

        Returns:
            (judgment, reason)
        """
        lines = response_text.strip().split('\n', 1)

        # 提取判断
        first_line = lines[0].strip().upper()
        if "DANGER" in first_line:
            judgment = "DANGER"
        elif "SAFE" in first_line:
            judgment = "SAFE"
        else:
            judgment = "UNCERTAIN"

        # 提取理由
        reason = lines[1].strip() if len(lines) > 1 else "无具体理由"

        return judgment, reason

    def judge(
        self,
        image: np.ndarray,
        expert_id: str,
        track_id: int,
        target_bbox: List[float],
        temporal_info: Optional[Dict[str, Any]] = None,
        zone_info: Optional[Dict[str, Any]] = None
    ) -> DangerJudgment:
        """
        判断目标是否危险

        Args:
            image: 原始图像（BGR格式）
            expert_id: 专家ID
            track_id: 跟踪ID
            target_bbox: 目标边界框 [x1, y1, x2, y2]
            temporal_info: 时序信息（如持续时间、状态变化等）
            zone_info: 区域信息（zone_name, zone_type等）

        Returns:
            DangerJudgment 判断结果
        """
        # 检查模型是否加载
        if self.model is None or self.processor is None:
            return DangerJudgment(
                judgment="UNCERTAIN",
                confidence=0.0,
                reason="VLM模型未加载",
                track_id=track_id
            )

        # 加载专家配置
        expert_config = self._load_expert_config(expert_id)
        if expert_config is None:
            return DangerJudgment(
                judgment="UNCERTAIN",
                confidence=0.0,
                reason="专家配置缺失",
                track_id=track_id
            )

        # 在图像上绘制目标框（红色标记）
        img_with_box = self._draw_target_box(image, target_bbox)

        # 转换图像格式（BGR → RGB → PIL Image）
        img_rgb = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # 构建Prompt（传入结构化信息）
        prompt = self._build_prompt(expert_config, temporal_info, zone_info)

        try:
            # 准备消息（Qwen-VL格式）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)

            # 准备模型输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            # 推理
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            # 解码响应
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 解析响应
            judgment, reason = self._parse_vlm_response(response_text)

            # 计算置信度（简化版，可以根据响应内容调整）
            confidence = 0.8 if judgment != "UNCERTAIN" else 0.3

            return DangerJudgment(
                judgment=judgment,
                confidence=confidence,
                reason=reason,
                track_id=track_id,
                raw_response=response_text
            )

        except Exception as e:
            print(f"[DangerJudge] VLM推理失败: {e}")
            return DangerJudgment(
                judgment="UNCERTAIN",
                confidence=0.0,
                reason=f"推理异常: {str(e)[:50]}",
                track_id=track_id
            )

    def batch_judge(
        self,
        image: np.ndarray,
        expert_id: str,
        targets: List[Dict[str, Any]]
    ) -> List[DangerJudgment]:
        """
        批量判断多个目标

        Args:
            image: 原始图像
            expert_id: 专家ID
            targets: 目标列表，每个包含 {track_id, bbox, temporal_info}

        Returns:
            判断结果列表
        """
        results = []
        for target in targets:
            result = self.judge(
                image=image,
                expert_id=expert_id,
                track_id=target["track_id"],
                target_bbox=target["bbox"],
                temporal_info=target.get("temporal_info")
            )
            results.append(result)

        return results

    # ==================== 异步推理接口 ====================

    def submit_async(
        self,
        key: VLMTaskKey,
        frame_id: int,
        image: np.ndarray,
        expert_id: str,
        track_id: int,
        target_bbox: List[float],
        temporal_info: Optional[Dict[str, Any]] = None,
        zone_info: Optional[Dict[str, Any]] = None,
        alert_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        提交异步VLM判断任务

        Args:
            key: 任务唯一标识 (expert_id, obj_id, rule_name)
            frame_id: 提交时的帧ID
            alert_context: 告警上下文（用于VLM完成后立即触发告警）
            其他参数同 judge()

        Returns:
            是否成功提交（如果已有相同key的任务在执行则返回False）
        """
        with self._lock:
            if key in self._pending_tasks:
                return False

            # 复制图像，因为原图可能被后续帧覆盖
            image_copy = image.copy()

            future = self._executor.submit(
                self.judge,
                image_copy,
                expert_id,
                track_id,
                target_bbox,
                temporal_info,
                zone_info
            )

            self._pending_tasks[key] = AsyncVLMTask(
                key=key,
                future=future,
                submit_frame_id=frame_id,
                context=alert_context
            )
            return True

    def poll_completed(self) -> List[Tuple[VLMTaskKey, DangerJudgment, int, Optional[Dict]]]:
        """
        获取所有已完成的异步任务结果

        Returns:
            列表，每项为 (key, judgment, submit_frame_id, context)
        """
        completed = []

        with self._lock:
            done_keys = []
            for key, task in self._pending_tasks.items():
                if task.future.done():
                    try:
                        result = task.future.result()
                        completed.append((key, result, task.submit_frame_id, task.context))
                    except Exception as e:
                        # 任务失败，返回 UNCERTAIN
                        completed.append((
                            key,
                            DangerJudgment(
                                judgment="UNCERTAIN",
                                confidence=0.0,
                                reason=f"异步任务异常: {str(e)[:50]}",
                                track_id=key[1]
                            ),
                            task.submit_frame_id,
                            task.context
                        ))
                    done_keys.append(key)

            for key in done_keys:
                del self._pending_tasks[key]

        return completed

    def has_pending_task(self, key: VLMTaskKey) -> bool:
        """检查是否有指定key的待处理任务"""
        with self._lock:
            return key in self._pending_tasks

    def cancel_task(self, key: VLMTaskKey) -> bool:
        """取消指定任务（如果尚未开始执行）"""
        with self._lock:
            task = self._pending_tasks.get(key)
            if task and task.future.cancel():
                del self._pending_tasks[key]
                return True
            return False

    def clear_pending(self) -> None:
        """清除所有待处理任务"""
        with self._lock:
            for task in self._pending_tasks.values():
                task.future.cancel()
            self._pending_tasks.clear()

    def shutdown(self) -> None:
        """关闭线程池"""
        self.clear_pending()
        self._executor.shutdown(wait=False)
