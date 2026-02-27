"""
第2层：专家检测层
加载并运行专家模型，返回检测结果
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .types import Detection, BBox


class ExpertDetector:
    """专家模型检测器"""

    def __init__(self, expert_id: str, models_dir: Path, device: Optional[str] = None):
        """
        初始化专家检测器

        Args:
            expert_id: 专家 ID
            models_dir: 模型目录（server/models/）
            device: 设备 (cpu | cuda | cuda:0 | 0)，默认自动选择
        """
        self.expert_id = expert_id
        self.model_dir = models_dir / expert_id

        # 加载专家元信息
        info_path = self.model_dir / "expert_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"专家元信息文件不存在: {info_path}")

        with open(info_path, 'r', encoding='utf-8') as f:
            self.info = json.load(f)

        # 加载模型
        model_path = self.model_dir / f"{expert_id}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"专家模型文件不存在: {model_path}")

        self.model = YOLO(str(model_path))
        self.expected_classes = self.info.get('expected_classes', [])

        # 移动模型到指定设备
        if device is not None:
            self.model.to(device)
            print(f"✓ 已加载专家: {expert_id}")
            print(f"  - 类别: {', '.join(self.expected_classes)}")
            print(f"  - 模型: {model_path}")
            print(f"  - 设备: {device}")
        else:
            print(f"✓ 已加载专家: {expert_id}")
            print(f"  - 类别: {', '.join(self.expected_classes)}")
            print(f"  - 模型: {model_path}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.6,  # 提高阈值到0.6，过滤误检（误检0.4-0.45，正确车辆0.8-0.9）
        iou_threshold: float = 0.5,    # NMS 阈值
        imgsz: int = 640               # 标准检测尺寸
    ) -> List[Detection]:
        """
        在单帧图像上运行检测

        Args:
            frame: 输入图像 (numpy array, BGR 格式)
            conf_threshold: 置信度阈值（降低以检测远距离小目标）
            iou_threshold: NMS IOU 阈值
            imgsz: 推理图像尺寸（应与训练时一致）

        Returns:
            检测结果列表
        """
        # 运行推理
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,      # 指定推理尺寸
            verbose=False
        )

        detections = []

        # 解析结果
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # 提取信息
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 获取类别名
                cls_name = result.names[cls_id]

                detections.append(Detection(
                    cls_name=cls_name,
                    conf=conf,
                    bbox=(x1, y1, x2, y2)
                ))

        return detections

    def get_expected_classes(self) -> List[str]:
        """获取专家期望检测的类别列表"""
        return self.expected_classes

    def get_scene_ids(self) -> List[str]:
        """获取专家适用的场景列表"""
        return self.info.get('scene_ids', [])
