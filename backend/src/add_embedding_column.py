#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
為 summaries 表添加 embedding 欄位
此腳本會：
1. 檢查 embedding 欄位是否存在
2. 如果不存在，則添加該欄位
"""

import sys
import os
from pathlib import Path

# 設置工作目錄和 Python 路徑
script_dir = Path(__file__).parent
app_dir = script_dir.parent
sys.path.insert(0, str(app_dir))
os.chdir(str(app_dir))

from sqlalchemy import text
from src.database import engine

def add_embedding_column():
    """為 summaries 表添加 embedding 欄位"""
    try:
        with engine.connect() as conn:
            # 檢查欄位是否已存在
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'summaries' 
                AND column_name = 'embedding'
            """)
            result = conn.execute(check_query).fetchone()
            
            if result:
                print("✓ embedding 欄位已存在，無需添加")
                return
            
            # 添加 embedding 欄位
            print("正在添加 embedding 欄位...")
            alter_query = text("""
                ALTER TABLE summaries 
                ADD COLUMN embedding vector(384)
            """)
            conn.execute(alter_query)
            conn.commit()
            print("✓ embedding 欄位添加成功")
            
    except Exception as e:
        print(f"❌ 添加 embedding 欄位失敗: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("="*60)
    print("PostgreSQL Embedding 欄位添加工具")
    print("="*60)
    print()
    
    add_embedding_column()
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)

