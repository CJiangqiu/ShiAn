"""
第3层：对象跟踪层
使用 ByteTrack 跟踪多帧中的对象，维护持久 ID
"""

from collections import defaultdict, deque
from typing import List, Dict, Optional

import numpy as np

from .types import Detection, TrackedObject, BBox
from .detector import ExpertDetector


class ObjectTracker:
    """对象跟踪器"""

    def __init__(self, config: dict):
        """
        初始化跟踪器

        Args:
            config: 跟踪配置
        """
        self.config = config
        self.track_buffer = config.get('track_buffer', 30)

        # 跟踪历史：obj_id → deque of bboxes
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.track_buffer)
        )

    def track(
        self,
        expert: ExpertDetector,
        frame: np.ndarray,
        conf_threshold: float = 0.6,  # 提高阈值，过滤误检（正确车辆0.8-0.9）
        iou_threshold: float = 0.5,    # NMS 阈值
        imgsz: int = 960,              # 使用训练时的图像尺寸
        device: Optional[str] = None,
        half: Optional[bool] = None,
        batch_size: Optional[int] = None
    ) -> List[TrackedObject]:
        """
        在单帧上执行检测 + 跟踪

        Args:
            expert: 专家检测器
            frame: 输入帧
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU 阈值
            imgsz: 推理图像尺寸

        Returns:
            跟踪对象列表
        """
        # 使用 YOLO 内置的 track() 方法（ByteTrack）
        results = expert.model.track(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,      # 指定推理尺寸
            device=device,
            half=half,
            batch=batch_size,
            persist=True,     # 保持跟踪器状态
            verbose=False
        )

        tracked_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # 提取信息
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (x1, y1, x2, y2)

                # 获取跟踪 ID
                if box.id is not None:
                    obj_id = int(box.id[0].item())
                else:
                    # 如果没有跟踪 ID，使用检测索引（fallback）
                    obj_id = -1

                # 获取类别名
                cls_name = result.names[cls_id]

                # 更新跟踪历史
                if obj_id != -1:
                    self.track_history[obj_id].append(bbox)
                    history = list(self.track_history[obj_id])
                else:
                    history = [bbox]

                tracked_objects.append(TrackedObject(
                    obj_id=obj_id,
                    cls_name=cls_name,
                    bbox=bbox,
                    conf=conf,
                    history=history
                ))

        return tracked_objects

    def get_history(self, obj_id: int) -> List[BBox]:
        """获取对象的历史轨迹"""
        return list(self.track_history.get(obj_id, []))

    def clear_history(self, obj_id: int):
        """清除对象的历史轨迹"""
        if obj_id in self.track_history:
            del self.track_history[obj_id]

    def reset(self):
        """重置跟踪器状态"""
        self.track_history.clear()
