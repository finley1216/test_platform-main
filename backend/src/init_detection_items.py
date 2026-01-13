#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化偵測項目的腳本
根據原有的 frame_prompt.md 創建預設的偵測項目
"""

from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, DetectionItem
from datetime import datetime

# 預設的偵測項目（從原有的 frame_prompt.md 提取）
DEFAULT_DETECTION_ITEMS = [
    {
        "name": "water_flood",
        "name_en": "water_flood",
        "name_zh": "水災",
        "description": "車輛明顯濺水 / 標線被水覆蓋 / 大片連續積水",
        "is_enabled": True,
    },
    {
        "name": "fire",
        "name_en": "fire",
        "name_zh": "火災",
        "description": "可見火焰或持續濃煙竄出",
        "is_enabled": True,
    },
    {
        "name": "abnormal_attire_face_cover_at_entry",
        "name_en": "abnormal_attire_face_cover_at_entry",
        "name_zh": "異常著裝/遮臉入場",
        "description": "門禁/閘機畫面中，臉被硬質裝備（如安全帽）遮擋仍嘗試通行",
        "is_enabled": True,
    },
    {
        "name": "person_fallen_unmoving",
        "name_en": "person_fallen_unmoving",
        "name_zh": "人員倒地不起",
        "description": "有人躺/倒於地面，且連續兩張以上影格姿勢不變",
        "is_enabled": True,
    },
    {
        "name": "double_parking_lane_block",
        "name_en": "double_parking_lane_block",
        "name_zh": "併排停車/車道阻塞",
        "description": "車道/出入口並排兩輛以上造成通行縮減/受阻",
        "is_enabled": True,
    },
    {
        "name": "smoking_outside_zone",
        "name_en": "smoking_outside_zone",
        "name_zh": "非管制區吸菸",
        "description": "手持燃燒香菸與煙霧，且明顯不在吸菸區標示內",
        "is_enabled": True,
    },
    {
        "name": "crowd_loitering",
        "name_en": "crowd_loitering",
        "name_zh": "聚眾逗留",
        "description": "同位置 ≥3 人在連續影格位置基本不變或樓梯旁多人閒坐（單張影像不足則 false）",
        "is_enabled": True,
    },
    {
        "name": "security_door_tamper",
        "name_en": "security_door_tamper",
        "name_zh": "突破安全門",
        "description": "反覆拉門把/推門縫/對鎖孔操作或操作「安全門/禁止進入」之門",
        "is_enabled": True,
    },
]


def init_detection_items(db: Session):
    """初始化偵測項目"""
    print("=== 偵測項目初始化 ===\n")
    
    # 檢查是否已有資料
    existing_count = db.query(DetectionItem).count()
    if existing_count > 0:
        print(f"⚠️  資料庫中已有 {existing_count} 個偵測項目")
        response = input("是否要清空並重新初始化？(y/N): ")
        if response.lower() != 'y':
            print("❌ 取消初始化")
            return
        
        # 清空現有資料
        db.query(DetectionItem).delete()
        db.commit()
        print("✓ 已清空現有偵測項目\n")
    
    # 創建預設偵測項目
    print(f"正在創建 {len(DEFAULT_DETECTION_ITEMS)} 個預設偵測項目...\n")
    
    for item_data in DEFAULT_DETECTION_ITEMS:
        item = DetectionItem(**item_data)
        db.add(item)
        print(f"✓ {item_data['name_zh']} ({item_data['name_en']})")
    
    db.commit()
    
    print(f"\n✓ 成功創建 {len(DEFAULT_DETECTION_ITEMS)} 個偵測項目")
    print("\n=== 下一步 ===")
    print("1. 請啟動後端服務")
    print("2. 使用前端的「偵測項目管理」介面來管理項目")
    print("3. 系統會自動根據啟用的項目更新 frame_prompt.md")


def main():
    """主函數"""
    # 創建資料表（如果不存在）
    Base.metadata.create_all(bind=engine)
    
    # 創建資料庫 session
    db = SessionLocal()
    
    try:
        init_detection_items(db)
    except Exception as e:
        print(f"\n❌ 初始化失敗：{e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
