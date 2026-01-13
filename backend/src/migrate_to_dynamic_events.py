#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料庫遷移腳本：從固定的事件 Boolean 欄位遷移到動態事件記錄
"""

import json
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Summary, DetectionItem

# 舊事件欄位與新的對應關係
OLD_EVENT_FIELDS = {
    "water_flood": "水災",
    "fire": "火災",
    "abnormal_attire_face_cover_at_entry": "異常著裝/遮臉入場",
    "person_fallen_unmoving": "人員倒地不起",
    "double_parking_lane_block": "併排停車/車道阻塞",
    "smoking_outside_zone": "非管制區吸菸",
    "crowd_loitering": "聚眾逗留",
    "security_door_tamper": "突破安全門",
}


def check_old_columns_exist(db: Session) -> bool:
    """檢查舊的 Boolean 欄位是否還存在"""
    try:
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'summaries' 
            AND column_name IN ('water_flood', 'fire', 'abnormal_attire_face_cover_at_entry')
        """))
        return len(result.fetchall()) > 0
    except Exception as e:
        print(f"❌ 檢查欄位時發生錯誤：{e}")
        return False


def check_new_columns_exist(db: Session) -> bool:
    """檢查新的動態事件欄位是否已存在"""
    try:
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'summaries' 
            AND column_name IN ('events_en', 'events_zh', 'events_json')
        """))
        return len(result.fetchall()) == 3
    except Exception as e:
        print(f"❌ 檢查欄位時發生錯誤：{e}")
        return False


def add_new_columns(db: Session):
    """添加新的動態事件欄位"""
    print("\n=== 步驟 1：添加新欄位 ===\n")
    
    try:
        db.execute(text("""
            ALTER TABLE summaries 
            ADD COLUMN IF NOT EXISTS events_en TEXT,
            ADD COLUMN IF NOT EXISTS events_zh TEXT,
            ADD COLUMN IF NOT EXISTS events_json TEXT
        """))
        db.commit()
        print("✓ 成功添加 events_en, events_zh, events_json 欄位")
    except Exception as e:
        print(f"❌ 添加欄位失敗：{e}")
        db.rollback()
        raise


def migrate_existing_data(db: Session):
    """遷移現有資料：將舊的 Boolean 欄位轉換為新的動態事件欄位"""
    print("\n=== 步驟 2：遷移現有資料 ===\n")
    
    # 查詢所有需要遷移的記錄
    summaries = db.query(Summary).all()
    total = len(summaries)
    
    if total == 0:
        print("⚠️  沒有需要遷移的資料")
        return
    
    print(f"找到 {total} 筆記錄需要遷移\n")
    
    migrated = 0
    skipped = 0
    
    for idx, summary in enumerate(summaries, 1):
        try:
            # 收集偵測到的事件
            events_en = []
            events_zh = []
            events_list = []
            
            # 檢查每個舊欄位
            for field_en, field_zh in OLD_EVENT_FIELDS.items():
                try:
                    # 嘗試讀取舊欄位的值
                    value = getattr(summary, field_en, None)
                    if value is True:
                        events_en.append(field_en)
                        events_zh.append(field_zh)
                        events_list.append({
                            "name_en": field_en,
                            "name_zh": field_zh,
                            "detected": True
                        })
                except AttributeError:
                    # 如果欄位不存在，跳過
                    continue
            
            # 只有在檢測到事件時才更新
            if events_en:
                summary.events_en = ", ".join(events_en)
                summary.events_zh = ", ".join(events_zh)
                summary.events_json = json.dumps(events_list, ensure_ascii=False)
                migrated += 1
                
                if migrated % 100 == 0:
                    print(f"進度：{migrated}/{total} ({migrated*100//total}%)")
            else:
                skipped += 1
        
        except Exception as e:
            print(f"❌ 遷移記錄 {summary.id} 時發生錯誤：{e}")
            continue
    
    try:
        db.commit()
        print(f"\n✓ 成功遷移 {migrated} 筆記錄")
        print(f"⚠️  跳過 {skipped} 筆無事件的記錄")
    except Exception as e:
        print(f"\n❌ 提交變更失敗：{e}")
        db.rollback()
        raise


def remove_old_columns(db: Session):
    """移除舊的 Boolean 欄位（可選）"""
    print("\n=== 步驟 3：移除舊欄位（可選）===\n")
    
    response = input("⚠️  是否要移除舊的 Boolean 欄位？這個操作不可逆！(y/N): ")
    
    if response.lower() != 'y':
        print("❌ 保留舊欄位")
        return
    
    try:
        for field in OLD_EVENT_FIELDS.keys():
            try:
                db.execute(text(f"ALTER TABLE summaries DROP COLUMN IF EXISTS {field}"))
                print(f"✓ 移除欄位：{field}")
            except Exception as e:
                print(f"⚠️  無法移除欄位 {field}：{e}")
        
        db.commit()
        print("\n✓ 成功移除所有舊欄位")
    except Exception as e:
        print(f"\n❌ 移除欄位失敗：{e}")
        db.rollback()
        raise


def create_detection_items_table(db: Session):
    """創建 detection_items 表（如果不存在）"""
    print("\n=== 步驟 4：創建 detection_items 表 ===\n")
    
    try:
        # 使用 SQLAlchemy 的 create_all 來創建表
        Base.metadata.create_all(bind=engine)
        print("✓ detection_items 表已就緒")
    except Exception as e:
        print(f"❌ 創建表失敗：{e}")
        raise


def main():
    """主函數"""
    print("=" * 60)
    print("資料庫遷移：從固定事件欄位到動態事件管理")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # 檢查資料庫狀態
        print("\n=== 檢查資料庫狀態 ===\n")
        
        has_old_columns = check_old_columns_exist(db)
        has_new_columns = check_new_columns_exist(db)
        
        print(f"舊欄位存在：{'是' if has_old_columns else '否'}")
        print(f"新欄位存在：{'是' if has_new_columns else '否'}")
        
        # 根據狀態決定遷移策略
        if not has_old_columns and has_new_columns:
            print("\n✓ 資料庫已經是新版本，無需遷移")
            response = input("\n是否要創建 detection_items 表？(y/N): ")
            if response.lower() == 'y':
                create_detection_items_table(db)
            return
        
        if not has_old_columns and not has_new_columns:
            print("\n⚠️  這是一個全新的資料庫")
            add_new_columns(db)
            create_detection_items_table(db)
            print("\n✓ 資料庫初始化完成")
            return
        
        # 執行完整遷移
        print("\n開始執行遷移...")
        response = input("確定要繼續嗎？(y/N): ")
        
        if response.lower() != 'y':
            print("❌ 取消遷移")
            return
        
        # 步驟 1：添加新欄位
        if not has_new_columns:
            add_new_columns(db)
        else:
            print("\n=== 步驟 1：跳過（新欄位已存在）===")
        
        # 步驟 2：遷移資料
        migrate_existing_data(db)
        
        # 步驟 3：移除舊欄位（可選）
        remove_old_columns(db)
        
        # 步驟 4：創建 detection_items 表
        create_detection_items_table(db)
        
        print("\n" + "=" * 60)
        print("✓ 遷移完成！")
        print("=" * 60)
        
        print("\n下一步：")
        print("1. 執行 init_detection_items.py 來創建預設的偵測項目")
        print("2. 啟動後端服務")
        print("3. 使用前端介面管理偵測項目")
        
    except Exception as e:
        print(f"\n❌ 遷移失敗：{e}")
        print("\n請檢查錯誤訊息並聯繫技術支援")
    finally:
        db.close()


if __name__ == "__main__":
    main()
