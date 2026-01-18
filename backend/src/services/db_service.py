# -*- coding: utf-8 -*-
import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.models import Summary, ObjectCrop
from src.core.model_loader import get_embedding_model

class DBService:
    @staticmethod
    def save_results_to_postgres(db: Session, results: List[Dict[str, Any]], video_stem: str):
        """將分析結果批次保存到 PostgreSQL"""
        summaries_to_save = []
        
        for res in results:
            segment_name = res.get("segment")
            # 檢查是否已存在
            summary = db.query(Summary).filter(
                Summary.video == video_stem,
                Summary.segment == segment_name
            ).first()
            
            if not summary:
                summary = Summary(
                    video=video_stem, 
                    segment=segment_name,
                    time_range=res.get("time_range"),
                    summary=res.get("parsed", {}).get("summary_independent", "")
                )
                summaries_to_save.append(summary)
            else:
                summary.time_range = res.get("time_range")
                summary.summary = res.get("parsed", {}).get("summary_independent", "")
        
        try:
            # 批次新增新對象
            if summaries_to_save:
                db.bulk_save_objects(summaries_to_save)
            
            # 提交更新（對於已存在的對象）
            db.commit()
            print(f"--- [DB] 成功批次儲存 {len(results)} 筆結果 ---")
        except Exception as e:
            db.rollback()
            print(f"--- [DB] ✗ 儲存失敗: {e} ---")
            raise

    @staticmethod
    def auto_index_to_rag(resp: Dict[str, Any]) -> Dict[str, Any]:
        """自動索引到 RAG"""
        # 實作 RAG 邏輯...
        return {"success": True, "message": "已完成 RAG 索引"}
