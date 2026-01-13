#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration: Add alerts table
建立 alerts 表用於記錄真正發生的事件警報
"""

from sqlalchemy import create_engine, text
from src.config import config
from src.database import Base
from src.models import Alert
import sys

def migrate():
    """執行遷移：建立 alerts 表"""
    
    # 建立資料庫連接
    DATABASE_URL = config.get_database_url()
    engine = create_engine(DATABASE_URL)
    
    try:
        # 檢查 alerts 表是否已存在
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'alerts'
                );
            """))
            exists = result.scalar()
            
            if exists:
                print("✓ alerts 表已存在，跳過建立")
                return
        
        # 建立 alerts 表
        print("正在建立 alerts 表...")
        Base.metadata.create_all(bind=engine, tables=[Alert.__table__])
        print("✓ alerts 表建立成功")
        
        # 顯示表結構
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'alerts'
                ORDER BY ordinal_position;
            """))
            
            print("\n=== Alerts 表結構 ===")
            for row in result:
                print(f"  {row[0]:20} {row[1]:20} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")
        
        print("\n✓ 遷移完成！")
        print("\n說明：")
        print("  - Summary 表：記錄所有影片分析結果")
        print("  - Alert 表：只記錄真正發生事件的警報")
        print("  - 只有偵測到事件的影片才會在 Alert 表中建立記錄")
        
    except Exception as e:
        print(f"❌ 遷移失敗：{e}", file=sys.stderr)
        sys.exit(1)
    finally:
        engine.dispose()

if __name__ == "__main__":
    migrate()
