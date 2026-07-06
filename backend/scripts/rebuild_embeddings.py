#!/usr/bin/env python3
"""
重建所有 summaries 的 embedding（使用新的 build_embed_text 邏輯）
執行：python scripts/rebuild_embeddings.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import SessionLocal
from src.models import Summary
from src.main import get_embedding_model, build_embed_text


def rebuild_all():
    db = SessionLocal()
    model = get_embedding_model()
    if model is None:
        print("❌ 無法載入 embedding model")
        return

    total = db.query(Summary).filter(
        Summary.message.isnot(None),
        Summary.message != ""
    ).count()
    print(f"共 {total} 筆需要重建")

    updated = 0
    batch_size = 100
    id_rows = db.query(Summary.id).filter(
        Summary.message.isnot(None),
        Summary.message != ""
    ).order_by(Summary.id.asc()).all()
    all_ids = [r[0] for r in id_rows]

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        records = db.query(Summary).filter(Summary.id.in_(batch_ids)).all()
        for summary in records:
            events = {
                "fire": summary.fire,
                "water_flood": summary.water_flood,
                "abnormal_attire_face_cover_at_entry": summary.abnormal_attire_face_cover_at_entry,
                "person_fallen_unmoving": summary.person_fallen_unmoving,
                "double_parking_lane_block": summary.double_parking_lane_block,
                "smoking_outside_zone": summary.smoking_outside_zone,
                "crowd_loitering": summary.crowd_loitering,
                "security_door_tamper": summary.security_door_tamper,
                "violence": summary.violence,
                "dangerous_items": summary.dangerous_items,
            }

            embed_text = build_embed_text(
                message=summary.message,
                event_reason=summary.event_reason,
                events=events,
            )
            try:
                summary.embedding = model.encode(embed_text, normalize_embeddings=True).tolist()
                updated += 1
            except Exception as e:
                print(f"  ⚠️  id={summary.id} 失敗: {e}")
        db.commit()
        print(f"  進度 {updated}/{total}...")

    db.commit()
    db.close()
    print(f"✅ 完成，共更新 {updated} 筆")


if __name__ == "__main__":
    rebuild_all()
