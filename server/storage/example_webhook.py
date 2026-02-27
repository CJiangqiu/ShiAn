"""
Webhook 存储后端示例

通过 HTTP POST 将告警推送到外部系统。
"""

import base64
import json
from typing import Optional, Dict, Any

import cv2
import requests

from .base import StorageBackend, AlertRecord


class WebhookStorage(StorageBackend):
    """
    Webhook 存储后端

    Args:
        url: Webhook 接收地址
        headers: 自定义请求头（如认证信息）
        include_frame: 是否包含 base64 编码的帧图像
        timeout: 请求超时时间（秒）
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        include_frame: bool = False,
        timeout: float = 10.0
    ):
        self.url = url
        self.headers = headers or {}
        self.headers.setdefault("Content-Type", "application/json")
        self.include_frame = include_frame
        self.timeout = timeout

    def _record_to_dict(self, record: AlertRecord, event_type: str) -> Dict[str, Any]:
        """将告警记录转换为 JSON 可序列化的字典"""
        data = {
            "event": event_type,
            "alert_id": record.alert_id,
            "expert_id": record.expert_id,
            "source_id": record.source_id,
            "risk_type": record.risk_type,
            "risk_desc": record.risk_desc,
            "severity": record.severity,
            "confidence": record.confidence,
            "bbox": record.bbox,
            "target_objects": record.target_objects,
            "timestamp": record.timestamp
        }

        if self.include_frame and record.frame is not None:
            _, buffer = cv2.imencode('.jpg', record.frame)
            data["frame_base64"] = base64.b64encode(buffer).decode('utf-8')

        return data

    def _send(self, data: Dict[str, Any]) -> None:
        """发送 HTTP POST 请求"""
        try:
            response = requests.post(
                self.url,
                json=data,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as e:
            # 记录错误但不中断主流程
            print(f"[WebhookStorage] 发送失败: {e}")

    def on_alert_start(self, record: AlertRecord) -> None:
        """告警开始 - 推送 alert_start 事件"""
        data = self._record_to_dict(record, "alert_start")
        self._send(data)

    def on_alert_update(self, record: AlertRecord) -> None:
        """告警更新 - 推送 alert_update 事件"""
        data = self._record_to_dict(record, "alert_update")
        self._send(data)

    def on_alert_end(self, record: AlertRecord) -> None:
        """告警结束 - 推送 alert_end 事件"""
        data = self._record_to_dict(record, "alert_end")
        self._send(data)

    def close(self) -> None:
        """Webhook 无需关闭"""
        pass
