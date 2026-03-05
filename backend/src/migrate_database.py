#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料庫遷移腳本 - 重建/修復資料表（summaries, object_crops, detection_items, alerts）
PostgreSQL 資料清空後，在專案根目錄執行：
  docker compose exec backend python src/migrate_database.py
或本機（需能連到 Postgres）：
  cd backend && PYTHONPATH=. python src/migrate_database.py
"""
import os
import sys

# 確保 backend 根目錄在 path，才能 import src.database / src.models
_backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

def migrate_database():
    """執行資料庫遷移"""
    print("=" * 80)
    print("開始資料庫遷移...")
    print("=" * 80)
    
    try:
        from src.database import engine, Base, ensure_pgvector_extension
        from src.models import Summary, ObjectCrop, DetectionItem, Alert
        from sqlalchemy import text, inspect

        print("\n[0/4] 確保 pgvector 擴展...")
        ensure_pgvector_extension()

        print("\n[1/4] 連接到資料庫...")
        with engine.connect() as conn:
            # 檢查連接
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✓ 成功連接到 PostgreSQL")
            print(f"  版本: {version}")
        
        print("\n[2/4] 檢查現有表結構...")
        inspector = inspect(engine)
        
        # 檢查所有必要的表是否存在
        required_tables = ['summaries', 'detection_items', 'object_crops', 'alerts']
        missing_tables = [table for table in required_tables if not inspector.has_table(table)]
        
        if missing_tables:
            print(f"⚠️ 缺少以下表: {', '.join(missing_tables)}")
            print("創建所有缺失的表...")
            Base.metadata.create_all(bind=engine)
            print("✓ 所有表已創建")
            
            # 如果創建了 detection_items 表，需要初始化預設資料
            if 'detection_items' in missing_tables:
                print("\n初始化 detection_items 預設資料...")
                try:
                    import subprocess
                    cwd = os.environ.get("PROJECT_ROOT", _backend_root)
                    result = subprocess.run(
                        [sys.executable, "src/init_detection_items.py"],
                        cwd=cwd,
                        env={**os.environ, "PYTHONPATH": cwd},
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        print("✓ detection_items 預設資料初始化完成")
                    else:
                        print(f"⚠️ detection_items 初始化失敗: {result.stderr}")
                except Exception as e:
                    print(f"⚠️ detection_items 初始化失敗: {e}")
            
            # 檢查是否還需要添加欄位
            if 'summaries' not in missing_tables:
                # summaries 表已存在但可能缺少欄位，繼續檢查
                pass
            else:
                return
        
        # 檢查欄位是否存在
        columns = [col['name'] for col in inspector.get_columns('summaries')]
        print(f"  現有欄位數量: {len(columns)}")
        
        missing_columns = []
        required_columns = {
            # 事件偵測欄位（布林值）
            'water_flood': 'BOOLEAN DEFAULT FALSE',
            'fire': 'BOOLEAN DEFAULT FALSE',
            'abnormal_attire_face_cover_at_entry': 'BOOLEAN DEFAULT FALSE',
            'person_fallen_unmoving': 'BOOLEAN DEFAULT FALSE',
            'double_parking_lane_block': 'BOOLEAN DEFAULT FALSE',
            'smoking_outside_zone': 'BOOLEAN DEFAULT FALSE',
            'crowd_loitering': 'BOOLEAN DEFAULT FALSE',
            'security_door_tamper': 'BOOLEAN DEFAULT FALSE',
            'event_reason': 'TEXT',
            # YOLO 偵測結果
            'yolo_detections': 'TEXT',
            'yolo_object_count': 'TEXT',
            'yolo_crops_dir': 'VARCHAR(500)',
            'yolo_total_detections': 'INTEGER',
            'yolo_total_frames_processed': 'INTEGER',
        }
        
        for col, col_type in required_columns.items():
            if col not in columns:
                missing_columns.append((col, col_type))
                print(f"  ✗ 缺少欄位: {col} ({col_type})")
        
        if not missing_columns:
            print("✓ 所有必要欄位都存在")
            return
        
        print(f"\n[3/4] 添加 {len(missing_columns)} 個缺少的欄位...")
        with engine.begin() as conn:
            for col_name, col_type in missing_columns:
                try:
                    sql = f"ALTER TABLE summaries ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
                    print(f"  執行: {sql}")
                    conn.execute(text(sql))
                    print(f"  ✓ 已添加欄位: {col_name}")
                except Exception as e:
                    print(f"  ✗ 添加欄位 {col_name} 失敗: {e}")
        
        print("\n[4/4] 驗證表結構...")
        inspector = inspect(engine)
        columns_after = [col['name'] for col in inspector.get_columns('summaries')]
        print(f"  更新後欄位數量: {len(columns_after)}")
        
        # 檢查是否所有欄位都存在
        all_exist = all(col in columns_after for col, _ in missing_columns)
        if all_exist:
            print("✓ 所有欄位都已成功添加")
        else:
            print("⚠️ 部分欄位添加失敗，請檢查日誌")
        
        print("\n" + "=" * 80)
        print("資料庫遷移完成！")
        print("=" * 80)
        print("\n建議操作：")
        print("1. 重啟後端容器: docker-compose restart backend")
        print("2. 檢查後端日誌: docker logs -f test_platform-main-backend-1")
        print("")
        
    except Exception as e:
        print(f"\n✗ 遷移失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    migrate_database()
