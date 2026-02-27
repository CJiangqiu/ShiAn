"""
区域配置管理（支持多边形）
按 source_id 保存/加载区域
"""
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
import json


class ZoneManager:
    """区域配置管理器"""

    def __init__(self, zones_dir: Path):
        self.zones_dir = Path(zones_dir)
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, List]] = {}

    def _source_key(self, source_id: str) -> str:
        source_id = source_id or "default"
        base_name = Path(source_id).stem or "source"
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", base_name)
        digest = hashlib.md5(source_id.encode("utf-8")).hexdigest()[:8]
        return f"{safe}_{digest}"

    def _zone_path(self, source_id: str) -> Path:
        return self.zones_dir / f"{self._source_key(source_id)}.yaml"

    def load_zones(self, source_id: str) -> Dict[str, List]:
        """加载指定 source 的区域配置"""
        source_id = source_id or "default"
        if source_id in self._cache:
            return self._cache[source_id]

        path = self._zone_path(source_id)
        if not path.exists():
            self._cache[source_id] = {}
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        zones = {}
        for zone in data.get("zones", []):
            name = zone.get("name")
            # 支持多边形 (points) 或矩形 (rect)
            points = zone.get("points") or zone.get("rect")
            if name and points:
                zones[name] = points

        self._cache[source_id] = zones
        return zones

    def save_zones(self, source_id: str, zones: Dict[str, List]) -> None:
        """保存指定 source 的区域配置"""
        source_id = source_id or "default"
        path = self._zone_path(source_id)

        payload = {
            "source_id": source_id,
            "zones": [
                {"name": name, "points": points}
                for name, points in zones.items()
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

        # 更新缓存
        self._cache[source_id] = zones

    def clear_cache(self, source_id: str = None) -> None:
        """清除缓存"""
        if source_id:
            self._cache.pop(source_id, None)
        else:
            self._cache.clear()

    def get_missing_zones(self, source_id: str, required_names: List[str]) -> List[str]:
        zones = self.load_zones(source_id)
        return [name for name in required_names if name not in zones]

    def point_in_zone(self, source_id: str, zone_name: str, x: int, y: int) -> bool:
        """判断点是否在区域内（多边形射线法）"""
        zones = self.load_zones(source_id)
        points = zones.get(zone_name)
        if not points:
            return False

        # 如果是矩形 [x1, y1, x2, y2]
        if len(points) == 4 and isinstance(points[0], (int, float)):
            x1, y1, x2, y2 = points
            return x1 <= x <= x2 and y1 <= y <= y2

        # 多边形射线法
        n = len(points)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = points[i][0], points[i][1]
            xj, yj = points[j][0], points[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
