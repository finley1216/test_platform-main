# -*- coding: utf-8 -*-
"""
資料庫遷移腳本：為 summaries 表添加 video 欄位
執行方式: docker compose exec backend python3 -m src.add_video_column
"""
import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.database import engine
from sqlalchemy import text

def add_video_column():
    """為 summaries 表添加 video 欄位"""
    print("=" * 60)
    print("為 summaries 表添加 video 欄位")
    print("=" * 60)
    
    try:
        with engine.connect() as conn:
            # 檢查欄位是否已存在
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'summaries' 
                AND column_name = 'video'
            """)
            result = conn.execute(check_query)
            exists = result.fetchone() is not None
            
            if exists:
                print("✓ video 欄位已存在，無需添加")
            else:
                # 添加 video 欄位
                alter_query = text("""
                    ALTER TABLE summaries 
                    ADD COLUMN video VARCHAR(255)
                """)
                conn.execute(alter_query)
                conn.commit()
                print("✓ 成功添加 video 欄位")
            
            # 顯示表結構
            print("\n當前 summaries 表結構:")
            desc_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'summaries'
                ORDER BY ordinal_position
            """)
            result = conn.execute(desc_query)
            for row in result:
                print(f"  - {row[0]}: {row[1]} (nullable: {row[2]})")
            
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("遷移完成")
    print("=" * 60)
    return True

if __name__ == "__main__":
    add_video_column()

