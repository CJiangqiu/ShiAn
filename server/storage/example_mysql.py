"""
MySQL 存储后端示例

依赖: pymysql, dbutils

使用前需创建表:
    CREATE TABLE alerts (
        id VARCHAR(64) PRIMARY KEY,
        expert_id VARCHAR(64) NOT NULL,
        source_id VARCHAR(64) NOT NULL,
        risk_type VARCHAR(64) NOT NULL,
        risk_desc TEXT,
        severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
        confidence FLOAT NOT NULL,
        bbox JSON,
        target_objects JSON,
        frame_path VARCHAR(255),
        started_at DATETIME NOT NULL,
        ended_at DATETIME,
        status ENUM('active', 'ended') DEFAULT 'active',
        INDEX idx_source (source_id),
        INDEX idx_expert (expert_id),
        INDEX idx_time (started_at)
    );
"""

import json
import os
from datetime import datetime
from typing import Optional

from .base import StorageBackend, AlertRecord


class MySQLStorage(StorageBackend):
    """
    MySQL 存储后端

    Args:
        host: 数据库主机
        port: 数据库端口
        user: 用户名
        password: 密码
        database: 数据库名
        frame_save_dir: 帧图像保存目录（可选）
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "shian",
        frame_save_dir: Optional[str] = None
    ):
        import pymysql
        from dbutils.pooled_db import PooledDB

        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        self.frame_save_dir = frame_save_dir
        if frame_save_dir:
            os.makedirs(frame_save_dir, exist_ok=True)

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

        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO alerts
                    (id, expert_id, source_id, risk_type, risk_desc, severity,
                     confidence, bbox, target_objects, frame_path, started_at, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                """
                cursor.execute(sql, (
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
                    datetime.fromtimestamp(record.timestamp)
                ))
            conn.commit()
        finally:
            conn.close()

    def on_alert_update(self, record: AlertRecord) -> None:
        """告警更新 - 更新置信度"""
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE alerts SET confidence = %s WHERE id = %s"
                cursor.execute(sql, (record.confidence, record.alert_id))
            conn.commit()
        finally:
            conn.close()

    def on_alert_end(self, record: AlertRecord) -> None:
        """告警结束 - 标记结束时间"""
        conn = self.pool.connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    UPDATE alerts
                    SET ended_at = %s, status = 'ended'
                    WHERE id = %s
                """
                cursor.execute(sql, (
                    datetime.fromtimestamp(record.timestamp),
                    record.alert_id
                ))
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """关闭连接池"""
        self.pool.close()
