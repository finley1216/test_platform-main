# -*- coding: utf-8 -*-
"""
資料庫遷移腳本：創建 object_crops 表
執行方式: docker compose exec backend python3 -m src.create_object_crop_table
"""
import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.database import engine, ensure_pgvector_extension
from sqlalchemy import text

def create_object_crop_table():
    """創建 object_crops 表"""
    print("=" * 60)
    print("創建 object_crops 表（用於以圖搜圖）")
    print("=" * 60)
    
    try:
        # 確保 pgvector extension 存在
        ensure_pgvector_extension()
        
        with engine.connect() as conn:
            # 檢查表是否已存在
            check_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'object_crops'
            """)
            result = conn.execute(check_query)
            exists = result.fetchone() is not None
            
            if exists:
                print("✓ object_crops 表已存在，跳過創建")
            else:
                # 創建 object_crops 表
                create_query = text("""
                    CREATE TABLE object_crops (
                        id SERIAL PRIMARY KEY,
                        summary_id INTEGER NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
                        crop_path VARCHAR(500),
                        label VARCHAR(50),
                        score REAL,
                        timestamp REAL,
                        frame INTEGER,
                        box TEXT,
                        clip_embedding vector(512),
                        reid_embedding vector(2048),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(create_query)
                conn.commit()  # 先提交表的創建
                
                # 創建索引
                index_queries = [
                    text("CREATE INDEX idx_object_crops_summary_id ON object_crops(summary_id)"),
                    text("CREATE INDEX idx_object_crops_label ON object_crops(label)"),
                    text("CREATE INDEX idx_object_crops_created_at ON object_crops(created_at)"),
                    # pgvector 索引（用於高效相似度搜索）
                    text("CREATE INDEX idx_object_crops_clip_embedding ON object_crops USING ivfflat (clip_embedding vector_cosine_ops) WITH (lists = 100)"),
                    # ReID embedding 索引跳過（2048 維超過 ivfflat 限制）
                    # text("CREATE INDEX idx_object_crops_reid_embedding ON object_crops USING ivfflat (reid_embedding vector_cosine_ops) WITH (lists = 100)"),
                ]
                
                for idx_query in index_queries:
                    try:
                        conn.execute(idx_query)
                        conn.commit()
                    except Exception as e:
                        print(f"  ⚠️  創建索引時出現警告（可能已存在）: {e}")
                
                print("✓ 成功創建 object_crops 表及索引")
            
            # 顯示表結構
            print("\n當前 object_crops 表結構:")
            desc_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'object_crops'
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
    create_object_crop_table()

