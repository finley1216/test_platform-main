#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
為現有的 PostgreSQL 記錄生成 embedding
此腳本會：
1. 查詢所有沒有 embedding 的記錄
2. 使用 SentenceTransformer 生成 embedding
3. 更新資料庫記錄
"""

import sys
import os
from pathlib import Path

# 設置工作目錄和 Python 路徑
# 在 Docker 容器中，工作目錄是 /app，腳本在 /app/src/generate_embeddings.py
script_dir = Path(__file__).parent  # /app/src
app_dir = script_dir.parent  # /app

# 確保 /app 在 Python 路徑中
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# 設置環境變數
os.environ.setdefault("PYTHONPATH", str(app_dir))
os.chdir(str(app_dir))  # 切換到 /app 目錄

from sqlalchemy.orm import Session
from src.database import SessionLocal, engine
from src.models import Summary

# 直接導入 embedding 相關函數，避免導入 RAGStore
# 確保 /app 在路徑中
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMER = True
except ImportError as e:
    HAS_SENTENCE_TRANSFORMER = False
    print(f"❌ sentence-transformers 未安裝: {e}")

# Embedding 模型配置
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = None

def get_embedding_model():
    """Get or initialize the SentenceTransformer model (CPU mode only)"""
    global _embedding_model
    if _embedding_model is None and HAS_SENTENCE_TRANSFORMER:
        try:
            # 強制使用 CPU 模式，避免 GPU 資源競爭
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 隱藏 GPU，強制使用 CPU
            cache_dir = os.getenv("TRANSFORMERS_CACHE", "/root/.cache/huggingface")
            # 嘗試使用本地緩存路徑
            local_model_path = "/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "modules.json")):
                # 使用本地路徑載入
                _embedding_model = SentenceTransformer(local_model_path, device='cpu')
            else:
                # 使用模型名稱載入（會嘗試從緩存讀取）
                _embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    cache_folder=cache_dir,
                    device='cpu'  # 明確指定 CPU 模式
                )
            print(f"✓ SentenceTransformer 模型載入: {EMBEDDING_MODEL_NAME} (CPU Mode)")
        except Exception as e:
            print(f"⚠️  載入 SentenceTransformer 模型失敗: {e}")
            print(f"   提示：容器內無法連網，需要手動下載模型或使用本地緩存")
            print(f"   解決方案：")
            print(f"   1. 在主機上下載模型到本地")
            print(f"   2. 通過 volume 掛載到容器")
            print(f"   3. 或使用已緩存的模型")
    return _embedding_model

def generate_embeddings_for_existing_records(batch_size: int = 100):
    """
    為現有記錄生成 embedding
    
    Args:
        batch_size: 每批處理的記錄數量
    """
    db: Session = SessionLocal()
    
    try:
        # 檢查 embedding 模型是否可用
        embedding_model = get_embedding_model()
        if not embedding_model:
            print("❌ Embedding 模型不可用，請確保 sentence-transformers 已安裝")
            return
        
        print(f"✓ 使用模型: {EMBEDDING_MODEL_NAME}")
        print(f"✓ Embedding 維度: 384\n")
        
        # 查詢所有沒有 embedding 的記錄（且有 message）
        query = db.query(Summary).filter(
            Summary.message.isnot(None),
            Summary.message != "",
            Summary.embedding.is_(None)
        )
        
        total_count = query.count()
        print(f"找到 {total_count} 筆需要生成 embedding 的記錄\n")
        
        if total_count == 0:
            print("✓ 所有記錄都已有 embedding，無需處理")
            return
        
        # 分批處理
        processed = 0
        failed = 0
        
        for offset in range(0, total_count, batch_size):
            batch = query.offset(offset).limit(batch_size).all()
            
            if not batch:
                break
            
            print(f"處理批次 {offset // batch_size + 1} ({offset + 1} - {min(offset + batch_size, total_count)} / {total_count})...")
            
            # 準備文本列表
            texts = []
            records = []
            
            for record in batch:
                # 使用 message 作為 embedding 的文本
                text = record.message or ""
                if text:
                    texts.append(text)
                    records.append(record)
            
            if not texts:
                continue
            
            try:
                # 批量生成 embedding
                embeddings = embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # 更新資料庫
                for record, embedding in zip(records, embeddings):
                    try:
                        record.embedding = embedding.tolist()
                        processed += 1
                    except Exception as e:
                        print(f"  ⚠️  更新記錄 {record.id} 失敗: {e}")
                        failed += 1
                
                # 提交批次
                db.commit()
                print(f"  ✓ 已處理 {len(records)} 筆記錄")
                
            except Exception as e:
                print(f"  ❌ 批次處理失敗: {e}")
                db.rollback()
                failed += len(records)
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"✓ 完成！")
        print(f"  - 成功處理: {processed} 筆")
        print(f"  - 失敗: {failed} 筆")
        print(f"  - 總計: {total_count} 筆")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("="*60)
    print("PostgreSQL Embedding 生成工具")
    print("="*60)
    print()
    
    generate_embeddings_for_existing_records(batch_size=100)

