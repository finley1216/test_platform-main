# -*- coding: utf-8 -*-
"""
Database models using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
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

class ObjectCrop(Base):
    """ObjectCrop table model - 存儲 YOLO 偵測到的物件切片及其 CLIP embedding（用於以圖搜圖）"""
    __tablename__ = "object_crops"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey("summaries.id"), nullable=False, index=True)  # 關聯到 summaries 表
    crop_path = Column(String(500), nullable=True)  # 物件切片圖片路徑
    label = Column(String(50), nullable=True)  # 物件類別（例如 "person", "car"）
    score = Column(Float, nullable=True)  # 偵測信心分數
    timestamp = Column(Float, nullable=True)  # 時間戳（秒）
    frame = Column(Integer, nullable=True)  # 幀號
    box = Column(Text, nullable=True)  # 邊界框 JSON: [x1, y1, x2, y2]
    
    # CLIP embedding（512 維）用於以圖搜圖（找外表相似的物件）
    clip_embedding = Column(Vector(512), nullable=True) if HAS_PGVECTOR else Column(Text, nullable=True)
    
    # ReID embedding（2048 維）用於物件 re-identification（找同一個人/同一輛車）
    reid_embedding = Column(Vector(2048), nullable=True) if HAS_PGVECTOR else Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now, index=True)

