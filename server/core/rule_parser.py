"""
DSL 规则解析器
将配置中的规则字符串编译为可执行函数
"""
import re
from typing import Dict, List, Callable, Any, Optional

from .relations import (
    bbox_contains,
    bbox_center_in_rect,
    calc_distance,
    calc_edge_distance,
    calc_iou,
    is_above,
    is_left_of,
    is_right_of,
)
from .types import TrackedObject


class RuleParser:
    """DSL 规则解析器"""

    def __init__(self):
        # 预编译正则
        self._func_pattern = re.compile(r'(\w+)\s*\(([^)]*)\)')

    def compile(self, condition: str) -> Callable:
        """
        编译 DSL 条件字符串为可执行函数

        Args:
            condition: DSL 条件字符串，例如 "NOT nearby(SELF, helmet, 50)"

        Returns:
            可执行函数 (primary_obj, all_objects, relations_cache) -> bool
        """
        # 预处理：替换函数调用为占位符
        placeholders = {}
        placeholder_idx = 0

        def replace_func(match):
            nonlocal placeholder_idx
            func_name = match.group(1)
            args_str = match.group(2)
            placeholder = f"__FUNC_{placeholder_idx}__"
            placeholders[placeholder] = (func_name, args_str)
            placeholder_idx += 1
            return placeholder

        processed = self._func_pattern.sub(replace_func, condition)

        # 替换逻辑运算符为 Python 语法
        processed = processed.replace(' AND ', ' and ')
        processed = processed.replace(' OR ', ' or ')
        processed = processed.replace('NOT ', 'not ')

        def evaluator(
            primary_obj: TrackedObject,
            all_objects: List[TrackedObject],
            relations_cache: Dict
        ) -> bool:
            local_vars = {}
            for placeholder, (func_name, args_str) in placeholders.items():
                result = self._execute_function(
                    func_name,
                    args_str,
                    primary_obj,
                    all_objects,
                    relations_cache
                )
                local_vars[placeholder] = result

            try:
                return eval(processed, {"__builtins__": {}}, local_vars)
            except Exception as e:
                print(f"[RuleParser] 规则执行错误: {condition} -> {e}")
                return False

        return evaluator

    def _execute_function(
        self,
        func_name: str,
        args_str: str,
        primary_obj: TrackedObject,
        all_objects: List[TrackedObject],
        relations_cache: Dict
    ) -> Any:
        """执行内置函数"""
        args = [a.strip() for a in args_str.split(',') if a.strip()]

        def resolve_obj(name: str) -> Optional[TrackedObject]:
            if name == 'SELF':
                return primary_obj
            for obj in all_objects:
                if obj.cls_name == name:
                    return obj
            return None

        def resolve_obj_list(name: str) -> List[TrackedObject]:
            if name == 'SELF':
                return [primary_obj]
            return [obj for obj in all_objects if obj.cls_name == name]

        if func_name == 'nearby':
            obj_a = resolve_obj(args[0])
            target_type = args[1]
            radius = float(args[2])
            if obj_a is None:
                return False
            targets = resolve_obj_list(target_type)
            for t in targets:
                if calc_distance(obj_a.bbox, t.bbox) <= radius:
                    return True
            return False

        if func_name == 'distance':
            obj_a = resolve_obj(args[0])
            target_type = args[1]
            if obj_a is None:
                return float('inf')
            targets = resolve_obj_list(target_type)
            if not targets:
                return float('inf')
            return min(calc_distance(obj_a.bbox, t.bbox) for t in targets)

        if func_name == 'edge_distance':
            obj_a = resolve_obj(args[0])
            target_type = args[1]
            if obj_a is None:
                return float('inf')
            targets = resolve_obj_list(target_type)
            if not targets:
                return float('inf')
            return min(calc_edge_distance(obj_a.bbox, t.bbox) for t in targets)

        if func_name == 'iou':
            obj_a = resolve_obj(args[0])
            target_type = args[1]
            if obj_a is None:
                return 0.0
            targets = resolve_obj_list(target_type)
            if not targets:
                return 0.0
            return max(calc_iou(obj_a.bbox, t.bbox) for t in targets)

        if func_name == 'contains':
            obj_a = resolve_obj(args[0])
            target_type = args[1]
            if obj_a is None:
                return False
            targets = resolve_obj_list(target_type)
            for t in targets:
                if bbox_contains(obj_a.bbox, t.bbox):
                    return True
            return False

        if func_name == 'above':
            obj_a = resolve_obj(args[0])
            obj_b = resolve_obj(args[1])
            if obj_a is None or obj_b is None:
                return False
            return is_above(obj_a.bbox, obj_b.bbox)

        if func_name == 'below':
            obj_a = resolve_obj(args[0])
            obj_b = resolve_obj(args[1])
            if obj_a is None or obj_b is None:
                return False
            return is_above(obj_b.bbox, obj_a.bbox)

        if func_name == 'left_of':
            obj_a = resolve_obj(args[0])
            obj_b = resolve_obj(args[1])
            if obj_a is None or obj_b is None:
                return False
            return is_left_of(obj_a.bbox, obj_b.bbox)

        if func_name == 'right_of':
            obj_a = resolve_obj(args[0])
            obj_b = resolve_obj(args[1])
            if obj_a is None or obj_b is None:
                return False
            return is_right_of(obj_a.bbox, obj_b.bbox)

        if func_name == 'in_zone':
            obj = resolve_obj(args[0])
            zone_name = args[1] if len(args) > 1 else ""
            if obj is None or not zone_name:
                return False
            zones = relations_cache.get("zones", {})
            zone_data = zones.get(zone_name)
            if not zone_data:
                return False
            # 计算 bbox 中心点
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            # 判断是矩形还是多边形
            if len(zone_data) == 4 and isinstance(zone_data[0], (int, float)):
                # 矩形 [x1, y1, x2, y2]
                x1, y1, x2, y2 = zone_data
                return x1 <= cx <= x2 and y1 <= cy <= y2
            else:
                # 多边形 [[x1,y1], [x2,y2], ...] - 射线法
                n = len(zone_data)
                inside = False
                j = n - 1
                for i in range(n):
                    xi, yi = zone_data[i][0], zone_data[i][1]
                    xj, yj = zone_data[j][0], zone_data[j][1]
                    if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                return inside

        if func_name == 'exists':
            target_type = args[0]
            return len(resolve_obj_list(target_type)) > 0

        if func_name == 'count':
            target_type = args[0]
            return len(resolve_obj_list(target_type))

        if func_name == 'area':
            obj = resolve_obj(args[0])
            if obj is None:
                return 0
            x1, y1, x2, y2 = obj.bbox
            return (x2 - x1) * (y2 - y1)

        print(f"[RuleParser] 未知函数: {func_name}")
        return False
