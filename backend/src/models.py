# -*- coding: utf-8 -*-
"""
Database models using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from datetime import datetime
from src.database import Base

# Import pgvector extension
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    # Fallback: define a placeholder Vector type if pgvector is not installed
    from sqlalchemy import TypeDecorator
    class Vector(TypeDecorator):
        impl = Text
        cache_ok = True


class Summary(Base):
    """Summary table model"""
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    start_timestamp = Column(DateTime, default=datetime.now)
    end_timestamp = Column(DateTime, nullable=True)
    location = Column(String(50), nullable=True)
    camera = Column(String(50), nullable=True)
    message = Column(Text, nullable=True)
    # Additional fields from segment JSON
    video = Column(String(255), nullable=True)  # 影片名稱（例如 "fire_1", "fire_2"），用於區分不同影片的相同 segment
    segment = Column(String(255), nullable=True)  # segment filename (e.g., "segment_0000.mp4")
    time_range = Column(String(50), nullable=True)  # Original time range string (e.g., "00:00:00 - 00:00:08")
    duration_sec = Column(Float, nullable=True)  # Segment duration in seconds
    time_sec = Column(Float, nullable=True)  # Processing time in seconds
    # Event detection fields from frame_analysis.events
    water_flood = Column(Boolean, default=False, nullable=True)  # 水災
    fire = Column(Boolean, default=False, nullable=True)  # 火災
    abnormal_attire_face_cover_at_entry = Column(Boolean, default=False, nullable=True)  # 異常著裝/遮臉入場
    person_fallen_unmoving = Column(Boolean, default=False, nullable=True)  # 人員倒地不起
    double_parking_lane_block = Column(Boolean, default=False, nullable=True)  # 併排停車/車道阻塞
    smoking_outside_zone = Column(Boolean, default=False, nullable=True)  # 非管制區吸菸
    crowd_loitering = Column(Boolean, default=False, nullable=True)  # 聚眾逗留
    security_door_tamper = Column(Boolean, default=False, nullable=True)  # 突破安全門
    event_reason = Column(Text, nullable=True)  # Event detection reason
    # Vector embedding for semantic search (384 dimensions for paraphrase-multilingual-MiniLM-L12-v2)
    embedding = Column(Vector(384), nullable=True) if HAS_PGVECTOR else Column(Text, nullable=True)
    
    # YOLO 偵測結果（整合到 summaries 表，不需要單獨的表）
    yolo_detections = Column(Text, nullable=True)  # YOLO 偵測結果 JSON（包含所有偵測到的物件信息）
    yolo_object_count = Column(Text, nullable=True)  # 物件計數 JSON（例如 {"person": 5, "car": 2}）
    yolo_crops_dir = Column(String(500), nullable=True)  # 物件切片目錄路徑（例如 "segment/fire_299/yolo_output/object_crops"）
    yolo_total_detections = Column(Integer, nullable=True)  # 總偵測物件數
    yolo_total_frames_processed = Column(Integer, nullable=True)  # 處理的總幀數
    
    created_at = Column(DateTime, default=datetime.now, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# [DEPRECATED] ObjectCrop 表已不再使用，YOLO 結果現在整合到 summaries 表中
# 完全移除類別定義，避免 SQLAlchemy 嘗試創建表
# 如果需要向後兼容，可以在需要時動態創建，但現在完全移除

