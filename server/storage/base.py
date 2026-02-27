"""
存储后端抽象接口

客户通过继承 StorageBackend 实现自定义存储逻辑。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class AlertRecord:
    """
    告警记录 - 传递给存储后端的数据结构

    Attributes:
        alert_id: 告警唯一标识
        expert_id: 专家模型ID
        source_id: 视频源ID
        risk_type: 风险类型
        risk_desc: 风险描述
        severity: 严重级别 (low/medium/high/critical)
        confidence: 置信度 (0.0-1.0)
        bbox: 边界框 [x1, y1, x2, y2]
        target_objects: 相关目标对象ID列表
        timestamp: 时间戳
        frame: 告警帧图像 (BGR numpy数组，可选)
    """
    alert_id: str
    expert_id: str
    source_id: str
    risk_type: str
    risk_desc: str
    severity: str
    confidence: float
    bbox: Optional[List[float]]
    target_objects: List[int]
    timestamp: float
    frame: Optional[np.ndarray] = None


class StorageBackend(ABC):
    """
    存储后端抽象基类

    客户需要实现以下方法来完成自定义存储:
    - on_alert_start: 告警开始时调用
    - on_alert_update: 告警持续更新时调用（可选实现）
    - on_alert_end: 告警结束时调用

    使用示例:
        class MyStorage(StorageBackend):
            def on_alert_start(self, record: AlertRecord) -> None:
                # 存储到数据库
                db.insert(record)

            def on_alert_update(self, record: AlertRecord) -> None:
                # 更新持续时间等
                db.update(record.alert_id, confidence=record.confidence)

            def on_alert_end(self, record: AlertRecord) -> None:
                # 标记告警结束
                db.mark_ended(record.alert_id)

        # 注册到 Pipeline
        pipeline.set_storage(MyStorage())
    """

    @abstractmethod
    def on_alert_start(self, record: AlertRecord) -> None:
        """
        告警开始时调用

        Args:
            record: 告警记录，包含告警详情和可选的帧图像
        """
        pass

    def on_alert_update(self, record: AlertRecord) -> None:
        """
        告警更新时调用（告警持续中）

        默认不做任何操作，子类可按需覆盖。

        Args:
            record: 更新后的告警记录
        """
        pass

    @abstractmethod
    def on_alert_end(self, record: AlertRecord) -> None:
        """
        告警结束时调用

        Args:
            record: 告警记录
        """
        pass

    def close(self) -> None:
        """
        关闭存储后端，释放资源

        子类可按需覆盖，用于关闭数据库连接等。
        """
        pass


class NoOpStorage(StorageBackend):
    """
    空实现 - 不执行任何存储操作

    作为默认后端，当客户未配置存储时使用。
    """

    def on_alert_start(self, record: AlertRecord) -> None:
        pass

    def on_alert_end(self, record: AlertRecord) -> None:
        pass
