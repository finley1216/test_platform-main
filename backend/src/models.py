# -*- coding: utf-8 -*-
"""
Database models using SQLAlchemy
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from datetime import datetime
from src.database import Base


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
    created_at = Column(DateTime, default=datetime.now, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

