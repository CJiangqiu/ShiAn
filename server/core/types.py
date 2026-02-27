"""
视安 (ShiAn) 核心数据结构定义
定义了推理流程中所有层使用的数据类型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

# 边界框类型：(x1, y1, x2, y2)
BBox = Tuple[float, float, float, float]


# ==================== 检测层 ====================

@dataclass
class Detection:
    """单个检测框"""
    cls_name: str           # 类别名称
    conf: float             # 置信度
    bbox: BBox              # 边界框 (x1, y1, x2, y2)


# ==================== 跟踪层 ====================

@dataclass
class TrackedObject:
    """跟踪后的对象"""
    obj_id: int             # 对象跟踪 ID
    cls_name: str           # 类别名称
    bbox: BBox              # 当前帧边界框
    conf: float             # 检测置信度
    history: List[BBox]     # 历史轨迹（最近 N 帧的 bbox）


# ==================== 关系层 ====================

@dataclass
class StaticRelations:
    """静态关系（单帧几何关系）"""
    # 重叠关系：(obj_a_id, obj_b_id) → {iou, region}
    overlaps: Dict[Tuple[int, int], dict] = field(default_factory=dict)

    # 距离关系：(obj_a_id, obj_b_id) → distance
    distances: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # 位置关系：(obj_a_id, obj_b_id) → "above"/"below"/"left"/"right"/"inside"
    positions: Dict[Tuple[int, int], str] = field(default_factory=dict)


@dataclass
class DynamicRelations:
    """动态关系（多帧运动趋势）"""
    # 运动趋势：obj_id → "expanding"/"shrinking"/"approaching_X"/"leaving_X"
    trends: Dict[int, str] = field(default_factory=dict)

    # 移动速度：obj_id → speed (pixels/frame)
    speeds: Dict[int, float] = field(default_factory=dict)

    # 持续帧数：obj_id → duration
    durations: Dict[int, int] = field(default_factory=dict)


@dataclass
class Relations:
    """场景关系汇总"""
    objects: List[TrackedObject]    # 跟踪对象列表
    static: StaticRelations         # 静态关系
    dynamic: DynamicRelations       # 动态关系


# ==================== 融合层 ====================

@dataclass
class ObjectConfidence:
    """对象置信度状态（时序融合）"""
    obj_id: int             # 对象 ID
    expert_id: str          # 专家 ID
    frame_conf: float       # 当前帧规则融合后的置信度
    fused_conf: float       # 时序融合后的置信度
    frame_count: int        # 已积累帧数
    lost_frames: int        # 丢失帧数


# ==================== 报警层 ====================

@dataclass
class Alert:
    """危险状态（持续性）"""
    alert_id: str           # 唯一标识
    expert_id: str          # 触发的专家 ID
    risk_type: str          # 危险类型 ID（如 "no_helmet"）
    risk_desc: str          # 危险类型描述（如 "未佩戴安全帽"）
    severity: str           # 严重程度：critical / high / medium / low
    confidence: float       # 当前置信度
    target_objects: List[int]   # 涉及的对象 ID
    bbox: BBox              # 报警区域边界框

    # 状态信息
    status: str             # "active" | "resolved"
    start_frame: int        # 开始帧
    end_frame: Optional[int] = None     # 结束帧
    duration_frames: int = 0             # 持续帧数

    # Web 扩展字段
    stream_id: Optional[str] = None     # 视频流 ID
    stream_name: Optional[str] = None   # 视频流名称
    start_time: Optional[float] = None  # 开始时间（Unix 时间戳）
    end_time: Optional[float] = None    # 结束时间（Unix 时间戳）
    duration: float = 0.0               # 持续时长（秒）


@dataclass
class AlertEvent:
    """报警事件（给前端/通知系统）"""
    event_type: str         # "start" | "update" | "end"
    alert: Alert            # 报警详情
    timestamp: float        # 时间戳
    frame_id: int           # 帧 ID
    screenshot: Optional[str] = None    # 截图 base64（仅 start 事件填充）
