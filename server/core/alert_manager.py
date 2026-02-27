"""
第7层：报警管理层
使用状态模型 + 双阈值防抖管理报警生命周期
"""

import time
import uuid
from typing import Dict, List, Optional

from .types import Alert, AlertEvent, ObjectConfidence, BBox


class AlertManager:
    """报警管理器"""

    def __init__(self):
        """初始化报警管理器"""
        # 活跃报警：(expert_id, obj_id, risk_type) → Alert
        self.active_alerts: Dict[tuple, Alert] = {}

    def update(
        self,
        expert_id: str,
        expert_config: dict,
        fused_states: Dict[int, Dict[str, ObjectConfidence]],
        frame_id: int,
        stream_id: Optional[str] = None,
        stream_name: Optional[str] = None
    ) -> List[AlertEvent]:
        """
        更新报警状态

        Args:
            expert_id: 专家 ID
            expert_config: 专家规则配置
            fused_states: 融合后的对象置信度状态（obj_id → risk_type → ObjectConfidence）
            frame_id: 当前帧 ID
            stream_id: 视频流 ID（Web 场景）
            stream_name: 视频流名称（Web 场景）

        Returns:
            报警事件列表
        """
        # 构建 risk_type → config 的映射
        risk_type_configs = {}
        for risk_config in expert_config['risk_types']:
            risk_type_configs[risk_config['id']] = risk_config

        events = []

        # 收集当前帧的所有 (obj_id, risk_type)
        current_pairs = set()
        for obj_id, risk_states in fused_states.items():
            for risk_type in risk_states.keys():
                current_pairs.add((obj_id, risk_type))

        # 处理当前帧的对象
        for obj_id, risk_states in fused_states.items():
            for risk_type, state in risk_states.items():
                risk_config = risk_type_configs[risk_type]
                trigger_threshold = risk_config['threshold']['trigger']
                release_threshold = risk_config['threshold']['release']
                severity = risk_config['severity']
                desc = risk_config['desc']

                key = (expert_id, obj_id, risk_type)
                fused_conf = state.fused_conf

                if key in self.active_alerts:
                    # 已有报警
                    alert = self.active_alerts[key]
                    current_time = time.time()

                    if fused_conf < release_threshold:
                        # 释放报警
                        alert.status = "resolved"
                        alert.end_frame = frame_id
                        alert.duration_frames = frame_id - alert.start_frame
                        alert.end_time = current_time
                        if alert.start_time:
                            alert.duration = current_time - alert.start_time

                        events.append(AlertEvent(
                            event_type="end",
                            alert=alert,
                            timestamp=current_time,
                            frame_id=frame_id
                        ))

                        del self.active_alerts[key]

                    else:
                        # 更新报警
                        alert.confidence = fused_conf
                        alert.duration_frames = frame_id - alert.start_frame
                        if alert.start_time:
                            alert.duration = current_time - alert.start_time

                        events.append(AlertEvent(
                            event_type="update",
                            alert=alert,
                            timestamp=current_time,
                            frame_id=frame_id
                        ))

                else:
                    # 无报警
                    if fused_conf >= trigger_threshold:
                        # 创建新报警
                        current_time = time.time()
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            expert_id=expert_id,
                            risk_type=risk_type,
                            risk_desc=desc,
                            severity=severity,
                            confidence=fused_conf,
                            target_objects=[obj_id],
                            bbox=(0, 0, 0, 0),  # TODO: 从对象计算 bbox
                            status="active",
                            start_frame=frame_id,
                            duration_frames=0,
                            stream_id=stream_id,
                            stream_name=stream_name,
                            start_time=current_time,
                            duration=0.0
                        )

                        self.active_alerts[key] = alert

                        events.append(AlertEvent(
                            event_type="start",
                            alert=alert,
                            timestamp=time.time(),
                            frame_id=frame_id
                        ))

        # 处理消失的对象（自动释放报警）
        all_keys = list(self.active_alerts.keys())
        for key in all_keys:
            eid, oid, risk_type = key
            if eid != expert_id:
                continue

            if (oid, risk_type) not in current_pairs:
                # 对象消失，释放报警
                alert = self.active_alerts[key]
                current_time = time.time()
                alert.status = "resolved"
                alert.end_frame = frame_id
                alert.duration_frames = frame_id - alert.start_frame
                alert.end_time = current_time
                if alert.start_time:
                    alert.duration = current_time - alert.start_time

                events.append(AlertEvent(
                    event_type="end",
                    alert=alert,
                    timestamp=current_time,
                    frame_id=frame_id
                ))

                del self.active_alerts[key]

        return events

    def get_active_alerts(self, expert_id: Optional[str] = None) -> List[Alert]:
        """
        获取所有活跃报警

        Args:
            expert_id: 专家 ID（可选，不指定则返回所有）

        Returns:
            报警列表
        """
        if expert_id is None:
            return list(self.active_alerts.values())

        return [
            alert for (eid, _, _), alert in self.active_alerts.items()
            if eid == expert_id
        ]

    def reset(self):
        """重置报警状态"""
        self.active_alerts.clear()
