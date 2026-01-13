# -*- coding: utf-8 -*-
"""
Database models using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, Enum
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
    # Event detection fields - 動態事件記錄（對應 DetectionItem 的 name_en 和 name_zh）
    events_en = Column(Text, nullable=True)  # 偵測到的事件（英文名稱，以逗號分隔，例如 "fire, water_flood"）
    events_zh = Column(Text, nullable=True)  # 偵測到的事件（中文名稱，以逗號分隔，例如 "火災, 水災"）
    events_json = Column(Text, nullable=True)  # 偵測到的事件 JSON 格式（詳細資訊，例如 [{"name_en": "fire", "name_zh": "火災", "detected": true}]）
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

class DetectionItem(Base):
    """DetectionItem table model - 偵測項目管理（動態管理事件類型）"""
    __tablename__ = "detection_items"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)  # 唯一識別名稱（用於資料庫記錄）
    name_en = Column(String(100), nullable=False)  # 英文名稱（用於 prompt）
    name_zh = Column(String(100), nullable=False)  # 中文名稱（用於顯示）
    description = Column(Text, nullable=True)  # 偵測標準描述（用於 prompt）
    is_enabled = Column(Boolean, default=True)  # 是否啟用（只有啟用的事件會出現在 prompt 中）
    alert_count = Column(Integer, default=0)  # 警報數量（統計）
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Alert(Base):
    """Alert table model - 事件警報記錄（只記錄真正發生的事件）"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    summary_id = Column(Integer, ForeignKey("summaries.id"), nullable=True, index=True)  # 關聯到 summaries 表（如果是影片事件）
    detection_item_id = Column(Integer, ForeignKey("detection_items.id"), nullable=True, index=True)  # 關聯到偵測項目
    
    # 警報基本資訊
    alert_type = Column(String(50), nullable=False, index=True)  # 事件類型（例如 'fire', 'water_flood', 'camera_disconnect'）
    title = Column(String(200), nullable=False)  # 警報標題（例如 "偵測到工廠A區發生火災"）
    message = Column(Text, nullable=True)  # 警報詳細訊息
    
    # 位置與設備資訊
    location = Column(String(100), nullable=True)  # 發生位置（例如 "工廠A區"）
    camera = Column(String(50), nullable=True)  # 攝影機編號
    device_id = Column(String(50), nullable=True)  # 設備識別碼
    
    # 影片資訊（如果是影片事件）
    video = Column(String(255), nullable=True)  # 影片名稱
    segment = Column(String(255), nullable=True)  # 影片片段名稱
    timestamp = Column(DateTime, nullable=True)  # 事件發生時間
    
    # 嚴重程度與狀態
    severity = Column(Enum('low', 'medium', 'high', 'critical', name='alert_severity'), default='medium')
    is_read = Column(Boolean, default=False, index=True)  # 是否已讀
    
    # 時間戳
    created_at = Column(DateTime, default=datetime.now, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)