"""
主推理流程
基于规则引擎完成端到端推理
"""
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .types import AlertEvent, ObjectConfidence
from .alert_manager import AlertManager
from .api_guard import build_provider, LLMProvider
from .danger_judge import DangerJudge, VLMTaskKey
from .rule_engine import RuleEngine, CompiledRule
from .detector import ExpertDetector
from .state_buffer import StateBuffer, RuleTemporalState
from .tracker import ObjectTracker
from .zone_manager import ZoneManager

from server.storage import StorageBackend, NoOpStorage
from server.storage.base import AlertRecord


class Pipeline:
    """视安 (ShiAn) 推理流水线"""

    def __init__(
        self,
        models_dir: Path,
        config_dir: Path,
        zones_dir: Optional[Path] = None,
        camera_id: Optional[str] = None
    ):
        """
        初始化流水线

        Args:
            models_dir: 模型目录 (server/models/)
            config_dir: 配置目录 (server/config/)
            zones_dir: 区域配置目录 (server/config/zones/)，可选
            camera_id: 摄像头ID（保留参数）
        """
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        self.camera_id = camera_id

        print("=" * 60)
        print("视安 (ShiAn) 流水线初始化")
        print("=" * 60)

        # 加载配置
        self._load_config()

        self._tracking_cfg = self.config.get('tracking', {})
        self._trackers: Dict[str, ObjectTracker] = {}

        # zones目录：优先使用传入参数，否则使用 config/zones
        if zones_dir:
            self.zone_manager = ZoneManager(zones_dir)
        else:
            self.zone_manager = ZoneManager(self.config_dir / "zones")

        self._zone_warned_sources = set()
        self._expert_warned_sources = set()
        self._inference_cfg = self.config.get('inference', {}) or {}
        self._experts: Dict[str, ExpertDetector] = {}
        self._source_experts: Dict[str, str] = {}

        self._state_cfg = self.config.get('state_buffer', {}) or {}
        self._state_buffers: Dict[str, StateBuffer] = {}

        # 本地VLM条件初始化
        print(f"[Pipeline] 配置检查: vlm_check_enabled={self.vlm_check_enabled}, local_vlm_enabled={self.local_vlm_enabled}")
        if self.vlm_check_enabled and self.local_vlm_enabled:
            self.danger_judge = DangerJudge(
                models_dir=models_dir,
                device=self.local_vlm_config.get('device', 'cuda'),
                model_name=self.local_vlm_config.get('model', 'qwen3-vl-2b-instruct')
            )
            print(f"[Pipeline] 本地VLM已启用: {self.local_vlm_config.get('model')}")
        else:
            self.danger_judge = None
            if not self.vlm_check_enabled:
                print("[Pipeline] VLM复检总开关已关闭（纯规则判断）")
            else:
                print("[Pipeline] 本地VLM已禁用（local_vlm.enabled=false）")

        self._alert_managers: Dict[str, AlertManager] = {}

        # 存储后端（默认不存储）
        self._storage: StorageBackend = NoOpStorage()
        self._current_frame: Optional[np.ndarray] = None

        # 云端VLM条件初始化
        self._init_remote_vlm()

        self._current_source_key: str = "default"
        self.frame_id: int = 0
        self._frame_ids: Dict[str, int] = {}
        self._last_tracked: Dict[str, list] = {}
        self._rule_engines: Dict[str, RuleEngine] = {}
        self._vlm_decision_cache: Dict[Tuple[str, int, str], Dict[str, int]] = {}
        self._llm_decision_cache: Dict[Tuple[str, int, str], Dict[str, int]] = {}

        print("=" * 60)
        print("流水线初始化完成")
        print("=" * 60)

    def _load_config(self) -> None:
        """加载客户端配置"""
        # 默认值
        self.config = {}
        self.vlm_check_enabled = False
        self.local_vlm_enabled = False
        self.remote_vlm_enabled = False
        self.local_vlm_config = {}
        self.remote_vlm_config = {}
        self.vlm_prompts = {}
        self.vlm_recheck_interval = 5

        # 读取主配置
        main_config_path = self.config_dir / "core_config.yaml"
        if not main_config_path.exists():
            print(f"[Pipeline] 警告: 配置不存在: {main_config_path}")
            return

        with open(main_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

        # VLM复检总开关
        vlm_check = self.config.get('vlm_check', {}) or {}
        self.vlm_check_enabled = vlm_check.get('enabled', False)

        if not self.vlm_check_enabled:
            print("[Pipeline] VLM复检总开关已关闭")
            return

        # 读取VLM详细配置
        vlm_config_path = self.config_dir / "vlm_config.yaml"
        if not vlm_config_path.exists():
            print(f"[Pipeline] VLM配置不存在: {vlm_config_path}")
            return

        with open(vlm_config_path, 'r', encoding='utf-8') as f:
            vlm_config = yaml.safe_load(f) or {}

        # 本地VLM配置
        local_vlm = vlm_config.get('local_vlm', {}) or {}
        self.local_vlm_enabled = local_vlm.get('enabled', False)
        self.local_vlm_config = local_vlm
        self.vlm_recheck_interval = int(local_vlm.get('recheck_interval', 5))

        # 云端VLM配置
        remote_vlm = vlm_config.get('remote_vlm', {}) or {}
        self.remote_vlm_enabled = remote_vlm.get('enabled', False)
        self.remote_vlm_config = remote_vlm

        # 通用提示词
        self.vlm_prompts = vlm_config.get('prompts', {}) or {}

    def _init_remote_vlm(self) -> None:
        """初始化云端VLM"""
        self.llm_provider: Optional[LLMProvider] = None

        if not self.vlm_check_enabled or not self.remote_vlm_enabled:
            return

        try:
            # 构建provider配置
            provider_name = self.remote_vlm_config.get('provider', 'qwen_vl')
            api_keys = self.remote_vlm_config.get('api_keys', {}) or {}
            models = self.remote_vlm_config.get('models', {}) or {}

            provider_config = {
                'provider': provider_name,
                'timeout_s': self.remote_vlm_config.get('timeout_s', 15),
                'retry': self.remote_vlm_config.get('retry', 2),
            }

            # 注入API密钥和模型配置
            if provider_name in api_keys:
                provider_config[provider_name] = {
                    **api_keys.get(provider_name, {}),
                    **models.get(provider_name, {})
                }

            # 注入通用提示词
            provider_config['default_prompts'] = self.vlm_prompts

            self.llm_provider = build_provider(provider_config)
            print(f"[Pipeline] 云端VLM已启用: {provider_name}")
        except Exception as e:
            print(f"[Pipeline] 云端VLM初始化失败: {e}")
            self.llm_provider = None

    def set_storage(self, storage: StorageBackend) -> None:
        """
        设置存储后端

        Args:
            storage: 存储后端实例（继承自 StorageBackend）

        示例:
            from server.storage.sqlite import SQLiteStorage
            pipeline.set_storage(SQLiteStorage("alerts.db"))
        """
        self._storage = storage

    def _dispatch_storage_events(
        self,
        events: List[AlertEvent],
        source_id: str,
        frame: Optional[np.ndarray] = None
    ) -> None:
        """将告警事件分发到存储后端"""
        for event in events:
            alert = event.alert
            record = AlertRecord(
                alert_id=alert.alert_id,
                expert_id=alert.expert_id,
                source_id=source_id or "default",
                risk_type=alert.risk_type,
                risk_desc=alert.risk_desc,
                severity=alert.severity,
                confidence=alert.confidence,
                bbox=list(alert.bbox) if alert.bbox else None,
                target_objects=alert.target_objects,
                timestamp=event.timestamp,
                frame=frame if event.event_type == "start" else None
            )

            if event.event_type == "start":
                self._storage.on_alert_start(record)
            elif event.event_type == "update":
                self._storage.on_alert_update(record)
            elif event.event_type == "end":
                self._storage.on_alert_end(record)

    def set_source_expert(self, source_id: str, expert_id: str) -> None:
        """为指定 source 设置专家模型"""
        source_id = source_id or "default"
        self._source_experts[source_id] = expert_id

    def _get_source_expert_id(self, source_id: str) -> Optional[str]:
        source_id = source_id or "default"
        return self._source_experts.get(source_id)

    def _get_expert(self, expert_id: str) -> ExpertDetector:
        if expert_id not in self._experts:
            device = self._inference_cfg.get('device', 'cpu')
            self._experts[expert_id] = ExpertDetector(expert_id, self.models_dir, device=device)
        return self._experts[expert_id]

    def get_required_zone_names(self, source_id: str) -> List[str]:
        expert_id = self._get_source_expert_id(source_id)
        if not expert_id:
            return []
        expert = self._get_expert(expert_id)
        zone_cfg = expert.info.get("zone", {}) or {}
        if not zone_cfg.get("required", False):
            return []
        zone_name = zone_cfg.get("zone_name") or f"{expert_id}_zone"
        return [zone_name]

    def get_missing_zones(self, source_id: str) -> List[str]:
        required = self.get_required_zone_names(source_id)
        if not required:
            return []
        return self.zone_manager.get_missing_zones(source_id, required)

    def save_zones(self, source_id: str, zones: Dict[str, List[int]]) -> None:
        self.zone_manager.save_zones(source_id, zones)

    def _annotate_frame_for_llm(
        self,
        frame: np.ndarray,
        tracked_objects: List,
        target_obj_id: int
    ) -> np.ndarray:
        """绘制目标框供远程复检"""
        annotated = frame.copy()
        target_obj = None
        for obj in tracked_objects:
            if obj.obj_id == target_obj_id:
                target_obj = obj
                break

        if target_obj is None:
            return annotated

        for obj in tracked_objects:
            if obj.obj_id == target_obj_id:
                continue
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 1)

        x1, y1, x2, y2 = map(int, target_obj.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)

        label = ">>> TARGET <<<"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            (0, 0, 255),
            -1
        )
        cv2.putText(
            annotated, label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )

        hint = f"Please check the RED BOX object (obj_{target_obj_id})"
        cv2.putText(
            annotated, hint,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255), 2
        )
        return annotated

    def _apply_llm_guard(
        self,
        expert_id: str,
        expert_info: dict,
        tracked_objects: List,
        rule_configs: Dict[str, CompiledRule],
        candidates: Dict[int, Dict[str, ObjectConfidence]],
        frame: np.ndarray
    ) -> Dict[int, Dict[str, ObjectConfidence]]:
        """远程 LLM 复检（可选）"""
        if not self.remote_vlm_enabled or self.llm_provider is None:
            return candidates

        scene_ids = expert_info.get('scene_ids', [])
        scene_desc = ', '.join(scene_ids) if scene_ids else expert_id
        obj_map = {obj.obj_id: obj for obj in tracked_objects}

        filtered_states: Dict[int, Dict[str, ObjectConfidence]] = {}
        for obj_id, risk_states in candidates.items():
            target_obj = obj_map.get(obj_id)
            if target_obj is None:
                continue

            target_class = target_obj.cls_name
            kept_risks = {}
            for risk_type, state in risk_states.items():
                alert_key = (expert_id, obj_id, risk_type)
                if alert_key in self._get_alert_manager(self._current_source_key).active_alerts:
                    kept_risks[risk_type] = state
                    continue

                cache_key = (expert_id, obj_id, risk_type)
                cache_entry = self._llm_decision_cache.get(cache_key)
                if cache_entry is not None:
                    cached_decision = cache_entry.get("decision")
                    if cached_decision == "approve":
                        kept_risks[risk_type] = state
                        continue
                    if cached_decision == "deny":
                        continue
                    if cached_decision == "delay":
                        next_check = cache_entry.get("next_check_frame", 0)
                        if self.frame_id < next_check:
                            continue
                        del self._llm_decision_cache[cache_key]
                    else:
                        action = self.llm_guard_uncertain_action
                        if action == "pass":
                            kept_risks[risk_type] = state
                        elif action == "degrade":
                            ratio = max(0.0, min(1.0, float(self.llm_guard_degrade_ratio)))
                            degraded_state = ObjectConfidence(
                                obj_id=state.obj_id,
                                expert_id=state.expert_id,
                                frame_conf=state.frame_conf * ratio,
                                fused_conf=state.fused_conf * ratio,
                                frame_count=state.frame_count,
                                lost_frames=state.lost_frames
                            )
                            kept_risks[risk_type] = degraded_state
                        continue

                rule_cfg = rule_configs.get(risk_type)
                risk_desc = rule_cfg.desc if rule_cfg else risk_type

                annotated_frame = self._annotate_frame_for_llm(
                    frame, tracked_objects, obj_id
                )

                context = {
                    "scene_desc": scene_desc,
                    "target_class": target_class,
                    "expert_id": expert_id,
                    "risk_type": risk_type,
                    "risk_desc": risk_desc,
                    "fused_conf": state.fused_conf,
                    "frame_id": self.frame_id,
                    "obj_id": obj_id,
                }
                decision = self.llm_provider.infer(annotated_frame, context)

                if decision.decision == "approve":
                    self._llm_decision_cache[cache_key] = {"decision": "approve"}
                    kept_risks[risk_type] = state
                elif decision.decision == "deny":
                    self._llm_decision_cache[cache_key] = {"decision": "deny"}
                else:
                    action = self.llm_guard_uncertain_action
                    if action == "pass":
                        self._llm_decision_cache[cache_key] = {"decision": "uncertain"}
                        kept_risks[risk_type] = state
                    elif action == "degrade":
                        ratio = max(0.0, min(1.0, float(self.llm_guard_degrade_ratio)))
                        degraded_state = ObjectConfidence(
                            obj_id=state.obj_id,
                            expert_id=state.expert_id,
                            frame_conf=state.frame_conf * ratio,
                            fused_conf=state.fused_conf * ratio,
                            frame_count=state.frame_count,
                            lost_frames=state.lost_frames
                        )
                        self._llm_decision_cache[cache_key] = {"decision": "uncertain"}
                        kept_risks[risk_type] = degraded_state
                    elif action == "delay":
                        self._llm_decision_cache[cache_key] = {
                            "decision": "delay",
                            "next_check_frame": self.frame_id + self.llm_guard_delay_frames
                        }

            if kept_risks:
                filtered_states[obj_id] = kept_risks

        return filtered_states

    def _get_rule_engine(self, expert_id: str, expert_info: dict) -> RuleEngine:
        if expert_id not in self._rule_engines:
            self._rule_engines[expert_id] = RuleEngine(expert_info)
        return self._rule_engines[expert_id]

    def _get_state_buffer(self, source_id: str) -> StateBuffer:
        if source_id not in self._state_buffers:
            self._state_buffers[source_id] = StateBuffer(
                window_size=int(self._state_cfg.get('window_size', 25)),
                fps=int(self._state_cfg.get('fps', 5)),
                max_lost_frames=int(self._state_cfg.get('max_lost_frames', 30))
            )
        return self._state_buffers[source_id]

    def _get_tracker(self, source_id: str) -> ObjectTracker:
        if source_id not in self._trackers:
            self._trackers[source_id] = ObjectTracker(self._tracking_cfg)
        return self._trackers[source_id]

    def _get_alert_manager(self, source_id: str) -> AlertManager:
        if source_id not in self._alert_managers:
            self._alert_managers[source_id] = AlertManager()
        return self._alert_managers[source_id]

    @staticmethod
    def _build_temporal_info(rule_name: str, state: RuleTemporalState) -> dict:
        return {
            "duration": state.duration,
            "attribute_descriptions": [
                f"规则: {rule_name}",
                f"置信度: {state.fused_conf:.2f}",
                f"连续违规: {state.consecutive_frames} 帧"
            ]
        }

    def _confirm_with_vlm(
        self,
        expert_id: str,
        obj_id: int,
        rule_name: str,
        bbox: Tuple[float, float, float, float],
        state: RuleTemporalState,
        frame: np.ndarray,
        expert_config: Optional[dict] = None,
        alert_context: Optional[dict] = None
    ) -> Optional[bool]:
        """
        调用 VLM 进行确认，返回 True/False/None
        调度逻辑：
        - 本地VLM可用 → 使用本地VLM
        - 本地VLM不可用，云端API可用 → 使用云端API替代
        - 两者都不可用 → 跳过验证，返回True

        Args:
            alert_context: 告警上下文（异步模式下用于VLM完成后立即触发告警）
        """
        cache_key = (expert_id, obj_id, rule_name)
        cached = self._vlm_decision_cache.get(cache_key)

        # 缓存命中
        if cached and cached.get("decision") in ("approve", "deny"):
            return cached["decision"] == "approve"

        # 异步任务进行中
        if cached and cached.get("decision") == "pending":
            return None

        if cached and cached.get("decision") == "uncertain":
            if self.frame_id < cached.get("next_check_frame", 0):
                return None
            del self._vlm_decision_cache[cache_key]

        # 调度：优先本地VLM，其次云端API
        if self.local_vlm_enabled and self.danger_judge is not None:
            return self._do_local_vlm_confirm(
                expert_id, obj_id, rule_name, bbox, state, frame, cache_key, alert_context
            )

        if self.remote_vlm_enabled and self.llm_provider is not None:
            return self._do_remote_vlm_confirm(
                expert_id, obj_id, rule_name, bbox, state, frame, cache_key, expert_config
            )

        # 两者都不可用，跳过验证
        return True

    def _do_local_vlm_confirm(
        self,
        expert_id: str,
        obj_id: int,
        rule_name: str,
        bbox: Tuple[float, float, float, float],
        state: RuleTemporalState,
        frame: np.ndarray,
        cache_key: Tuple[str, int, str],
        alert_context: Optional[dict] = None
    ) -> Optional[bool]:
        """
        使用本地VLM进行确认（异步模式）

        Returns:
            True: 确认危险
            False: 确认安全
            None: 待定（已提交异步任务或任务进行中）
        """
        # 检查是否已有异步任务在执行
        vlm_key: VLMTaskKey = cache_key
        if self.danger_judge.has_pending_task(vlm_key):
            return None

        # 提交异步任务
        temporal_info = self._build_temporal_info(rule_name, state)
        submitted = self.danger_judge.submit_async(
            key=vlm_key,
            frame_id=self.frame_id,
            image=frame,
            expert_id=expert_id,
            track_id=obj_id,
            target_bbox=list(bbox),
            temporal_info=temporal_info,
            zone_info=None,
            alert_context=alert_context
        )

        if submitted:
            # 标记为待定，等待异步结果
            self._vlm_decision_cache[cache_key] = {
                "decision": "pending",
                "submit_frame_id": self.frame_id
            }

        return None

    def _do_remote_vlm_confirm(
        self,
        expert_id: str,
        obj_id: int,
        rule_name: str,
        bbox: Tuple[float, float, float, float],
        state: RuleTemporalState,
        frame: np.ndarray,
        cache_key: Tuple[str, int, str],
        expert_config: Optional[dict] = None
    ) -> Optional[bool]:
        """使用云端API替代本地VLM进行确认"""
        # 如果没有传入expert_config，尝试获取
        if expert_config is None:
            expert = self._experts.get(expert_id)
            expert_config = expert.info if expert else {}

        temporal_info = self._build_temporal_info(rule_name, state)
        decision = self.llm_provider.infer_as_vlm(
            image=frame,
            expert_config=expert_config,
            track_id=obj_id,
            target_bbox=list(bbox),
            temporal_info=temporal_info,
            zone_info=None
        )

        if decision.decision == "approve":
            self._vlm_decision_cache[cache_key] = {
                "decision": "approve",
                "frame_id": self.frame_id
            }
            return True

        if decision.decision == "deny":
            self._vlm_decision_cache[cache_key] = {
                "decision": "deny",
                "frame_id": self.frame_id
            }
            return False

        self._vlm_decision_cache[cache_key] = {
            "decision": "uncertain",
            "next_check_frame": self.frame_id + self.vlm_recheck_interval
        }
        return None

    @staticmethod
    def _build_alert_config(rule_configs: Dict[str, CompiledRule]) -> Dict[str, List[dict]]:
        risk_types = []
        for rule_name, rule_cfg in rule_configs.items():
            threshold = rule_cfg.threshold
            risk_types.append({
                "id": rule_name,
                "desc": rule_cfg.desc,
                "severity": rule_cfg.severity,
                "threshold": {
                    "trigger": threshold,
                    "release": max(0.1, threshold * 0.5)
                }
            })
        return {"risk_types": risk_types}

    def _process_async_vlm_results(self, source_id: Optional[str] = None) -> List[AlertEvent]:
        """处理已完成的异步VLM任务，立即触发告警"""
        if self.danger_judge is None:
            return []

        all_events = []
        completed = self.danger_judge.poll_completed()

        for vlm_key, judgment, submit_frame_id, context in completed:
            expert_id, obj_id, rule_name = vlm_key
            cache_key = vlm_key

            if judgment.judgment == "DANGER":
                self._vlm_decision_cache[cache_key] = {
                    "decision": "approve",
                    "frame_id": self.frame_id
                }
                # 立即触发告警
                if context:
                    ctx_source = context.get("stream_id") or self._current_source_key
                    events = self._get_alert_manager(ctx_source).update(
                        expert_id=context["expert_id"],
                        expert_config=context["expert_config"],
                        fused_states=context["fused_states"],
                        frame_id=self.frame_id,
                        stream_id=context.get("stream_id"),
                        stream_name=context.get("stream_name")
                    )
                    all_events.extend(events)

            elif judgment.judgment == "SAFE":
                self._vlm_decision_cache[cache_key] = {
                    "decision": "deny",
                    "frame_id": self.frame_id
                }

            else:  # UNCERTAIN
                self._vlm_decision_cache[cache_key] = {
                    "decision": "uncertain",
                    "next_check_frame": self.frame_id + self.vlm_recheck_interval
                }

        return all_events

    def process_frame(
        self,
        frame: np.ndarray,
        source_id: Optional[str] = None,
        stream_name: Optional[str] = None
    ) -> List[AlertEvent]:
        """
        处理单帧

        Args:
            frame: 输入图像 (BGR, numpy array)
            source_id: 视频源 ID（可作为 stream_id）
            stream_name: 视频流名称（Web 场景用于前端显示）

        Returns:
            报警事件列表
        """
        source_key = source_id or "default"
        self._current_source_key = source_key  # 供内部方法访问
        self._frame_ids.setdefault(source_key, 0)
        self._frame_ids[source_key] += 1
        frame_id = self._frame_ids[source_key]
        self.frame_id = frame_id  # 兼容VLM方法访问当前帧ID
        self._current_frame = frame  # 保存当前帧引用（用于drain时存储）

        # 处理异步VLM结果（可能产生延迟告警）
        async_events = self._process_async_vlm_results(source_id)

        expert_id = self._get_source_expert_id(source_key)
        if not expert_id:
            if source_key not in self._expert_warned_sources:
                print(f"[Pipeline] 未配置专家: {source_key}")
                self._expert_warned_sources.add(source_key)
            self._last_tracked[source_key] = []
            # 即使没有专家，也返回异步VLM产生的告警
            return async_events
        missing_zones = self.get_missing_zones(source_key)
        if missing_zones and source_key not in self._zone_warned_sources:
            print(f"[Zone] 缺少区域配置: {source_key} -> {', '.join(missing_zones)}")
            self._zone_warned_sources.add(source_key)
        all_events: List[AlertEvent] = []
        all_tracked_objects = []

        expert = self._get_expert(expert_id)
        tracker = self._get_tracker(source_key)
        tracked_objects = tracker.track(
            expert,
            frame,
            device=self._inference_cfg.get('device'),
            half=self._inference_cfg.get('half'),
            batch_size=self._inference_cfg.get('batch_size')
        )
        state_buffer = self._get_state_buffer(source_key)
        if not tracked_objects:
            state_buffer.update({}, frame_id)
            self._cleanup_vlm_cache(expert_id, set())
            self._cleanup_llm_guard_cache(expert_id, set())
            self._last_tracked[source_key] = []
            return async_events

        all_tracked_objects.extend(tracked_objects)
        current_obj_ids = {obj.obj_id for obj in tracked_objects}

        rule_engine = self._get_rule_engine(expert_id, expert.info)
        zones = self.zone_manager.load_zones(source_key)
        frame_results = rule_engine.evaluate(
            tracked_objects,
            relations_cache={"zones": zones}
        )

        temporal_states = state_buffer.update(frame_results, frame_id)
        if not temporal_states:
            self._cleanup_vlm_cache(expert_id, set())
            self._cleanup_llm_guard_cache(expert_id, current_obj_ids)
            self._last_tracked[source_key] = all_tracked_objects
            return async_events

        obj_map = {obj.obj_id: obj for obj in tracked_objects}
        rule_configs = rule_engine.get_rule_configs()
        alert_candidates: Dict[int, Dict[str, ObjectConfidence]] = {}
        active_keys = set()

        for obj_id, rule_states in temporal_states.items():
            for rule_name, state in rule_states.items():
                rule_cfg = rule_configs.get(rule_name)
                if rule_cfg is None:
                    continue

                if state.fused_conf < rule_cfg.threshold:
                    continue
                if state.duration < rule_cfg.min_duration:
                    continue

                key = (expert_id, obj_id, rule_name)
                active_keys.add(key)

                if rule_cfg.need_vlm_confirm:
                    target = obj_map.get(obj_id)
                    if target is None:
                        continue

                    # 构建告警上下文（用于异步VLM完成后立即触发告警）
                    obj_conf = ObjectConfidence(
                        obj_id=obj_id,
                        expert_id=expert_id,
                        frame_conf=1.0 if state.frame_violated else 0.0,
                        fused_conf=state.fused_conf,
                        frame_count=state.window_frames,
                        lost_frames=0
                    )
                    alert_context = {
                        "expert_id": expert_id,
                        "expert_config": self._build_alert_config(rule_configs),
                        "fused_states": {obj_id: {rule_name: obj_conf}},
                        "stream_id": source_id,
                        "stream_name": stream_name
                    }

                    decision = self._confirm_with_vlm(
                        expert_id,
                        obj_id,
                        rule_name,
                        target.bbox,
                        state,
                        frame,
                        expert_config=expert.info,
                        alert_context=alert_context
                    )
                    if decision is not True:
                        continue

                alert_candidates.setdefault(obj_id, {})[rule_name] = ObjectConfidence(
                    obj_id=obj_id,
                    expert_id=expert_id,
                    frame_conf=1.0 if state.frame_violated else 0.0,
                    fused_conf=state.fused_conf,
                    frame_count=state.window_frames,
                    lost_frames=0
                )

        if not alert_candidates:
            self._cleanup_vlm_cache(expert_id, active_keys)
            self._cleanup_llm_guard_cache(expert_id, current_obj_ids)
            self._last_tracked[source_key] = all_tracked_objects
            return async_events

        guarded_candidates = self._apply_llm_guard(
            expert_id,
            expert.info,
            tracked_objects,
            rule_configs,
            alert_candidates,
            frame
        )

        if not guarded_candidates:
            self._cleanup_vlm_cache(expert_id, active_keys)
            self._cleanup_llm_guard_cache(expert_id, current_obj_ids)
            self._last_tracked[source_key] = all_tracked_objects
            return async_events

        expert_config = self._build_alert_config(rule_configs)
        alert_manager = self._get_alert_manager(source_key)
        events = alert_manager.update(
            expert_id,
            expert_config,
            guarded_candidates,
            frame_id,
            stream_id=source_id,
            stream_name=stream_name
        )
        all_events.extend(events)
        all_events.extend(async_events)  # 加入异步VLM产生的告警

        # 分发到存储后端
        if all_events:
            self._dispatch_storage_events(all_events, source_id, frame)

        self._cleanup_vlm_cache(expert_id, active_keys)
        self._cleanup_llm_guard_cache(expert_id, current_obj_ids)

        self._last_tracked[source_key] = all_tracked_objects
        return all_events

    def _cleanup_vlm_cache(self, expert_id: str, active_keys: set) -> None:
        stale_keys = [
            key for key in self._vlm_decision_cache
            if key[0] == expert_id and key not in active_keys
        ]
        for key in stale_keys:
            del self._vlm_decision_cache[key]

    def _cleanup_llm_guard_cache(self, expert_id: str, current_obj_ids: set) -> None:
        stale_keys = [
            key for key in self._llm_decision_cache
            if key[0] == expert_id and key[1] not in current_obj_ids
        ]
        for key in stale_keys:
            del self._llm_decision_cache[key]

    def get_active_alerts(self, source_id: str = "default"):
        """获取指定源的活跃报警"""
        return self._get_alert_manager(source_id).get_active_alerts()

    def get_last_tracked_objects(self, source_id: str = "default"):
        """获取指定源最近一帧的跟踪对象（用于可视化）"""
        return self._last_tracked.get(source_id, [])

    def reset_source(self, source_id: str):
        """重置指定视频源的状态（不影响其他源）"""
        self._frame_ids.pop(source_id, None)
        self._last_tracked.pop(source_id, None)
        if source_id in self._trackers:
            self._trackers[source_id].reset()
            del self._trackers[source_id]
        if source_id in self._state_buffers:
            self._state_buffers[source_id].reset()
            del self._state_buffers[source_id]
        if source_id in self._alert_managers:
            self._alert_managers[source_id].reset()
            del self._alert_managers[source_id]

    def reset(self):
        """重置所有状态"""
        self._frame_ids.clear()
        self._last_tracked.clear()
        for t in self._trackers.values():
            t.reset()
        self._trackers.clear()
        for buffer in self._state_buffers.values():
            buffer.reset()
        self._state_buffers.clear()
        for am in self._alert_managers.values():
            am.reset()
        self._alert_managers.clear()
        self._vlm_decision_cache.clear()
        self._llm_decision_cache.clear()
        if self.danger_judge is not None:
            self.danger_judge.clear_pending()

    def drain_vlm_results(
        self,
        timeout: float = 120.0,
        poll_interval: float = 0.5,
        source_id: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> List[AlertEvent]:
        """
        等待所有异步VLM任务完成并处理结果

        适用于视频播放结束后，继续等待后台VLM推理完成

        Args:
            timeout: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            source_id: 视频源ID（用于存储）
            callback: 每次有新结果时的回调 callback(events: List[AlertEvent])

        Returns:
            所有告警事件列表
        """
        if self.danger_judge is None:
            return []

        all_events: List[AlertEvent] = []
        start_time = time.time()

        print(f"[Pipeline] 开始等待VLM任务完成，超时: {timeout}s")

        while time.time() - start_time < timeout:
            # 检查是否还有待处理任务
            pending_count = len(self.danger_judge._pending_tasks)
            if pending_count == 0:
                print("[Pipeline] 所有VLM任务已完成")
                break

            # 处理已完成的结果
            events = self._process_async_vlm_results(source_id)
            if events:
                all_events.extend(events)
                # 分发到存储后端
                self._dispatch_storage_events(events, source_id, self._current_frame)
                # 回调通知
                if callback:
                    callback(events)
                print(f"[Pipeline] VLM完成，产生 {len(events)} 个告警，剩余 {pending_count - 1} 个任务")

            # 等待一段时间再检查
            time.sleep(poll_interval)

        # 最后再检查一次
        final_events = self._process_async_vlm_results(source_id)
        if final_events:
            all_events.extend(final_events)
            self._dispatch_storage_events(final_events, source_id, self._current_frame)
            if callback:
                callback(final_events)

        remaining = len(self.danger_judge._pending_tasks)
        if remaining > 0:
            print(f"[Pipeline] 警告: 超时，仍有 {remaining} 个VLM任务未完成")

        return all_events

    def get_pending_vlm_count(self) -> int:
        """获取待处理的VLM任务数量"""
        if self.danger_judge is None:
            return 0
        return len(self.danger_judge._pending_tasks)
