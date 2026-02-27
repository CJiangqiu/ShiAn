"""
规则引擎
根据专家配置的规则判断危险状态
"""
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional

from .types import TrackedObject
from .rule_parser import RuleParser


@dataclass
class CompiledRule:
    """已编译规则"""
    func: Callable[[TrackedObject, List[TrackedObject], Dict], bool]
    condition: str
    min_duration: float
    severity: str
    need_vlm_confirm: bool
    threshold: float
    desc: str


class RuleEngine:
    """规则引擎"""

    MODE_PERSON_CENTRIC = 0
    MODE_OBJECT_CENTRIC = 1

    def __init__(self, expert_info: Dict[str, Any]):
        """
        初始化规则引擎

        Args:
            expert_info: 专家配置（从 expert_info.json 加载）
        """
        self.expert_id = expert_info.get('expert_id', 'unknown')
        self.detection_mode = expert_info.get('detection_mode', 0)
        self.primary_object = expert_info.get('primary_object', 'person')

        self._parser = RuleParser()
        self._compiled_rules: Dict[str, CompiledRule] = {}
        self._compile_rules(expert_info.get('alert_rules', {}))

        print(f"[RuleEngine] 初始化完成 | 专家: {self.expert_id}")
        print(f"            模式: {'人本' if self.detection_mode == 0 else '物本'}")
        print(f"            主体: {self.primary_object}")
        print(f"            规则数: {len(self._compiled_rules)}")

    def _compile_rules(self, rules_config: Dict[str, Dict]) -> None:
        """编译所有规则"""
        for rule_name, rule_cfg in rules_config.items():
            condition = rule_cfg.get('condition', '')
            if not condition:
                print(f"[RuleEngine] 警告: 规则 {rule_name} 缺少 condition")
                continue

            try:
                compiled_func = self._parser.compile(condition)
                threshold = rule_cfg.get('threshold', rule_cfg.get('confidence_threshold', 0.6))
                self._compiled_rules[rule_name] = CompiledRule(
                    func=compiled_func,
                    condition=condition,
                    min_duration=rule_cfg.get('min_duration', 3.0),
                    severity=rule_cfg.get('severity', 'medium'),
                    need_vlm_confirm=rule_cfg.get('need_vlm_confirm', False),
                    threshold=float(threshold),
                    desc=rule_cfg.get('desc', rule_name),
                )
                print(f"            - {rule_name}: {condition}")
            except Exception as e:
                print(f"[RuleEngine] 编译规则失败 {rule_name}: {e}")

    def _select_primary_objects(self, objects: List[TrackedObject]) -> List[TrackedObject]:
        """根据 detection_mode 选择遍历主体"""
        if self.detection_mode == self.MODE_PERSON_CENTRIC:
            target_classes = {"person"}
        else:
            primary = self.primary_object
            if isinstance(primary, list):
                target_classes = {str(item) for item in primary}
            else:
                target_classes = {str(primary)}
        return [obj for obj in objects if obj.cls_name in target_classes]

    def evaluate(
        self,
        objects: List[TrackedObject],
        relations_cache: Optional[Dict[str, Any]] = None
    ) -> Dict[int, Dict[str, bool]]:
        """
        评估所有规则

        Args:
            objects: 当前帧的跟踪对象列表

        Returns:
            {obj_id: {rule_name: violated(T/F)}}
        """
        results: Dict[int, Dict[str, bool]] = {}
        relations_cache = relations_cache or {}
        primary_objects = self._select_primary_objects(objects)

        for primary_obj in primary_objects:
            obj_results = {}
            for rule_name, rule_cfg in self._compiled_rules.items():
                try:
                    violated = rule_cfg.func(primary_obj, objects, relations_cache)
                except Exception as e:
                    print(f"[RuleEngine] 规则执行错误 {rule_name}: {e}")
                    violated = False
                obj_results[rule_name] = bool(violated)
            if obj_results:
                results[primary_obj.obj_id] = obj_results

        return results

    def get_rule_configs(self) -> Dict[str, CompiledRule]:
        """获取已编译的规则配置"""
        return self._compiled_rules
