"""
SQLite 存储后端示例

使用 Python 内置 sqlite3。
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

from .base import StorageBackend, AlertRecord


class SQLiteStorage(StorageBackend):
    """
    SQLite 存储后端

    Args:
        db_path: 数据库文件路径
        frame_save_dir: 帧图像保存目录（可选）
    """

    def __init__(
        self,
        db_path: str = "alerts.db",
        frame_save_dir: Optional[str] = None
    ):
        self.db_path = db_path
        self.frame_save_dir = frame_save_dir
        if frame_save_dir:
            os.makedirs(frame_save_dir, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    expert_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    risk_type TEXT NOT NULL,
                    risk_desc TEXT,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT,
                    target_objects TEXT,
                    frame_path TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON alerts(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expert ON alerts(expert_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON alerts(started_at)")
            conn.commit()
        finally:
            conn.close()

    def _save_frame(self, record: AlertRecord) -> Optional[str]:
        """保存帧图像，返回文件路径"""
        if record.frame is None or not self.frame_save_dir:
            return None

        import cv2
        filename = f"{record.alert_id}_{record.timestamp:.0f}.jpg"
        filepath = os.path.join(self.frame_save_dir, filename)
        cv2.imwrite(filepath, record.frame)
        return filepath

    def on_alert_start(self, record: AlertRecord) -> None:
        """告警开始 - 插入新记录"""
        frame_path = self._save_frame(record)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO alerts
                (id, expert_id, source_id, risk_type, risk_desc, severity,
                 confidence, bbox, target_objects, frame_path, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """, (
                record.alert_id,
                record.expert_id,
                record.source_id,
                record.risk_type,
                record.risk_desc,
                record.severity,
                record.confidence,
                json.dumps(record.bbox) if record.bbox else None,
                json.dumps(record.target_objects),
                frame_path,
                datetime.fromtimestamp(record.timestamp).isoformat()
            ))
            conn.commit()
        finally:
            conn.close()

    def on_alert_update(self, record: AlertRecord) -> None:
        """告警更新 - 更新置信度"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE alerts SET confidence = ? WHERE id = ?",
                (record.confidence, record.alert_id)
            )
            conn.commit()
        finally:
            conn.close()

    def on_alert_end(self, record: AlertRecord) -> None:
        """告警结束 - 标记结束时间"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE alerts
                SET ended_at = ?, status = 'ended'
                WHERE id = ?
            """, (
                datetime.fromtimestamp(record.timestamp).isoformat(),
                record.alert_id
            ))
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """SQLite 无需显式关闭连接池"""
        pass
