"""
存储后端模块

提供告警数据持久化的抽象接口。

示例实现:
    - example_mysql.py: MySQL 存储
    - example_sqlite.py: SQLite 存储
    - example_webhook.py: Webhook 推送
"""

from .base import StorageBackend, NoOpStorage

__all__ = ["StorageBackend", "NoOpStorage"]
