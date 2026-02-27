"""
关系计算函数
为规则引擎 DSL 提供基础几何关系能力
"""
import math
from typing import Tuple

from .types import BBox


def calc_distance(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算中心点距离（像素）"""
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b
    center_a = ((x1_a + x2_a) / 2, (y1_a + y2_a) / 2)
    center_b = ((x1_b + x2_b) / 2, (y1_b + y2_b) / 2)
    return math.sqrt(
        (center_a[0] - center_b[0]) ** 2 +
        (center_a[1] - center_b[1]) ** 2
    )


def calc_edge_distance(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算边缘最近距离（像素）"""
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b

    if x2_a < x1_b:
        dx = x1_b - x2_a
    elif x2_b < x1_a:
        dx = x1_a - x2_b
    else:
        dx = 0

    if y2_a < y1_b:
        dy = y1_b - y2_a
    elif y2_b < y1_a:
        dy = y1_a - y2_b
    else:
        dy = 0

    return math.sqrt(dx ** 2 + dy ** 2)


def calc_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算 IoU"""
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b

    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    """检查 outer 是否包含 inner"""
    x1_o, y1_o, x2_o, y2_o = outer
    x1_i, y1_i, x2_i, y2_i = inner
    return x1_o <= x1_i and y1_o <= y1_i and x2_i <= x2_o and y2_i <= y2_o


def is_above(bbox_a: BBox, bbox_b: BBox) -> bool:
    """判断 bbox_a 是否在 bbox_b 上方"""
    _, y1_a, _, y2_a = bbox_a
    _, y1_b, _, _ = bbox_b
    center_a_y = (y1_a + y2_a) / 2
    return center_a_y < y1_b


def is_left_of(bbox_a: BBox, bbox_b: BBox) -> bool:
    """判断 bbox_a 是否在 bbox_b 左侧"""
    x1_a, _, x2_a, _ = bbox_a
    x1_b, _, _, _ = bbox_b
    center_a_x = (x1_a + x2_a) / 2
    return center_a_x < x1_b


def is_right_of(bbox_a: BBox, bbox_b: BBox) -> bool:
    """判断 bbox_a 是否在 bbox_b 右侧"""
    x1_a, _, x2_a, _ = bbox_a
    _, _, x2_b, _ = bbox_b
    center_a_x = (x1_a + x2_a) / 2
    return center_a_x > x2_b


def bbox_center_in_rect(bbox: BBox, rect: list) -> bool:
    """判断 bbox 中心点是否在 rect 内"""
    if not rect or len(rect) != 4:
        return False
    x1, y1, x2, y2 = rect
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2
