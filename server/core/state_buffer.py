"""
时序融合
维护每个对象的规则状态，用于计算置信度与持续时间
"""
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, Deque, Optional


@dataclass
class RuleTemporalState:
    """规则时序状态输出"""
    obj_id: int
    rule_name: str
    frame_violated: bool
    fused_conf: float
    violated_frames: int
    window_frames: int
    consecutive_frames: int
    duration: float


class _RuleWindow:
    """规则滑动窗口"""
    def __init__(self, window_size: int):
        self.window: Deque[bool] = deque(maxlen=window_size)
        self.violated_frames = 0
        self.consecutive_frames = 0
        self._consecutive_start_time: Optional[float] = None

    def update(self, violated: bool) -> None:
        if len(self.window) == self.window.maxlen:
            oldest = self.window.popleft()
            if oldest:
                self.violated_frames -= 1

        self.window.append(violated)
        if violated:
            self.violated_frames += 1
            self.consecutive_frames += 1
            if self._consecutive_start_time is None:
                self._consecutive_start_time = time.time()
        else:
            self.consecutive_frames = 0
            self._consecutive_start_time = None

    def duration(self) -> float:
        """返回当前连续违规的实际持续时间（秒）"""
        if self._consecutive_start_time is None:
            return 0.0
        return time.time() - self._consecutive_start_time

    def window_size(self) -> int:
        return len(self.window)

    def confidence(self) -> float:
        size = len(self.window)
        if size == 0:
            return 0.0
        return self.violated_frames / size


class StateBuffer:
    """对象时序状态缓冲区"""

    def __init__(
        self,
        window_size: int = 25,
        fps: int = 5,
        max_lost_frames: int = 30
    ):
        """
        初始化状态缓冲区

        Args:
            window_size: 滑动窗口大小（帧数）
            fps: 视频帧率
            max_lost_frames: 对象消失超过该帧数则清理
        """
        self.window_size = window_size
        self.fps = fps
        self.max_lost_frames = max_lost_frames

        self._object_states: Dict[int, Dict[str, _RuleWindow]] = {}
        self._lost_frames: Dict[int, int] = {}

    def update(
        self,
        frame_results: Dict[int, Dict[str, bool]],
        frame_id: int
    ) -> Dict[int, Dict[str, RuleTemporalState]]:
        """
        更新时序状态

        Args:
            frame_results: {obj_id: {rule_name: violated}}
            frame_id: 当前帧 ID

        Returns:
            {obj_id: {rule_name: RuleTemporalState}}
        """
        temporal_results: Dict[int, Dict[str, RuleTemporalState]] = {}
        current_obj_ids = set(frame_results.keys())

        for obj_id, rule_results in frame_results.items():
            if obj_id not in self._object_states:
                self._object_states[obj_id] = {}
            self._lost_frames[obj_id] = 0

            for rule_name, violated in rule_results.items():
                rule_state = self._object_states[obj_id].get(rule_name)
                if rule_state is None:
                    rule_state = _RuleWindow(self.window_size)
                    self._object_states[obj_id][rule_name] = rule_state

                rule_state.update(violated)
                window_frames = rule_state.window_size()
                fused_conf = rule_state.confidence()
                duration = rule_state.duration()

                temporal_results.setdefault(obj_id, {})[rule_name] = RuleTemporalState(
                    obj_id=obj_id,
                    rule_name=rule_name,
                    frame_violated=violated,
                    fused_conf=fused_conf,
                    violated_frames=rule_state.violated_frames,
                    window_frames=window_frames,
                    consecutive_frames=rule_state.consecutive_frames,
                    duration=round(duration, 2)
                )

        self._cleanup_stale_objects(current_obj_ids)
        return temporal_results

    def _cleanup_stale_objects(self, current_obj_ids: set) -> None:
        """清理已消失对象"""
        stale_ids = []
        for obj_id in list(self._object_states.keys()):
            if obj_id in current_obj_ids:
                continue
            self._lost_frames[obj_id] = self._lost_frames.get(obj_id, 0) + 1
            if self._lost_frames[obj_id] > self.max_lost_frames:
                stale_ids.append(obj_id)

        for obj_id in stale_ids:
            del self._object_states[obj_id]
            if obj_id in self._lost_frames:
                del self._lost_frames[obj_id]

    def get_active_objects(self) -> list:
        """获取当前活跃对象 ID"""
        return list(self._object_states.keys())

    def reset(self) -> None:
        """重置状态缓冲区"""
        self._object_states.clear()
        self._lost_frames.clear()
