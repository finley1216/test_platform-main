#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料庫遷移腳本 - 重建/修復資料表（summaries, object_crops, detection_items, alerts）
PostgreSQL 資料清空後，在專案根目錄執行：
  docker compose exec backend python src/migrate_database.py
或本機（需能連到 Postgres）：
  cd backend && PYTHONPATH=. python src/migrate_database.py

補齊欄位後若物理欄位順序與 models.Summary 不一致（例如 ALTER 把新欄位加在表尾），
會以複製資料的方式重建 summaries，使欄位順序與 ORM 一致（violence / dangerous_items 在 event_reason 前）。
"""
import os
import sys

# 確保 backend 根目錄在 path，才能 import src.database / src.models
_backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)


def _embedding_sql_type(engine):
    """與 models.Summary 一致：有 pgvector 時用 vector(384)，否則 TEXT。"""
    from sqlalchemy import text

    try:
        from pgvector.sqlalchemy import Vector  # noqa: F401
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return "vector(384)"
    except Exception:
        return "TEXT"


def _summaries_schema_name(inspector) -> str:
    """目前連線預設 schema（多為 public）。"""
    s = getattr(inspector, "default_schema_name", None) or "public"
    return s


def _summaries_column_order_matches_model(engine, inspector, summary_table) -> bool:
    """
    以 information_schema.ordinal_position 為準（與 \\d 顯示順序一致），
    比對「實際表上屬於 Summary 模型的欄位」順序是否與 ORM 定義一致。
    """
    from sqlalchemy import text

    if not inspector.has_table("summaries"):
        return True
    schema = _summaries_schema_name(inspector)
    model_cols = [c.name for c in summary_table.columns]
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = 'summaries'
                ORDER BY ordinal_position
                """
            ),
            {"schema": schema},
        ).fetchall()
    actual = [r[0] for r in rows]
    actual_subset = [c for c in actual if c in model_cols]
    desired_subset = [c for c in model_cols if c in actual]
    return actual_subset == desired_subset


def _snapshot_foreign_keys_to_summaries(inspector):
    out = []
    for table_name in inspector.get_table_names():
        for fk in inspector.get_foreign_keys(table_name):
            if fk.get("referred_table") != "summaries":
                continue
            opts = fk.get("options") or {}
            ondel = opts.get("ondelete") or fk.get("ondelete")
            out.append(
                {
                    "table": table_name,
                    "name": fk["name"],
                    "constrained_columns": list(fk["constrained_columns"]),
                    "referred_columns": list(fk["referred_columns"]),
                    "ondelete": ondel,
                }
            )
    return out


def _snapshot_indexes_on_summaries(inspector):
    idxs = []
    for idx in inspector.get_indexes("summaries"):
        if idx.get("unique") and idx["name"] and "pkey" in idx["name"]:
            continue
        idxs.append(
            {
                "name": idx["name"],
                "column_names": list(idx["column_names"]),
                "unique": bool(idx.get("unique")),
            }
        )
    return idxs


def _reflect_summaries_table(engine):
    """反射現有 summaries 表（保留實際欄位型別，例如 embedding 為 TEXT 或 vector）。"""
    from sqlalchemy import MetaData, inspect

    insp = inspect(engine)
    schema = _summaries_schema_name(insp)
    meta = MetaData()
    meta.reflect(bind=engine, schema=schema, only=["summaries"], resolve_fks=False)
    for k in (f"{schema}.summaries", "summaries"):
        if k in meta.tables:
            return meta.tables[k]
    if meta.tables:
        for k, t in meta.tables.items():
            if getattr(t, "name", None) == "summaries":
                return t
    raise RuntimeError("無法反射 summaries 表")


def _reorder_summaries_table(engine, Summary):
    """
    以「反射自現有 summaries」的欄位型別建立 summaries_new（避免 ORM 與 DB 型別不一致導致 INSERT 失敗），
    欄位順序依 models.Summary，多出的欄位接在後面；複製資料後交換表名並還原 FK／索引。
    """
    from sqlalchemy import inspect, text
    from sqlalchemy.schema import CreateTable

    inspector = inspect(engine)
    if not inspector.has_table("summaries"):
        return False

    if _summaries_column_order_matches_model(engine, inspector, Summary.__table__):
        return False

    print(
        "\n  [summaries] 偵測到欄位物理順序與 models.Summary 不一致（常見於 ALTER 追加欄位），"
        "將依「現有欄位型別」複製並重建表以對齊順序…"
    )

    src = _reflect_summaries_table(engine)
    model_order = [c.name for c in Summary.__table__.columns]
    ordered_names = []
    for n in model_order:
        if n in src.columns:
            ordered_names.append(n)
    for n in src.columns.keys():
        if n not in ordered_names:
            ordered_names.append(n)

    new_cols = []
    for n in ordered_names:
        col = src.columns[n]
        try:
            new_cols.append(col.copy())
        except Exception:
            new_cols.append(col._copy())
    from sqlalchemy import MetaData, Table

    summaries_new = Table("summaries_new", MetaData(), *new_cols)

    fk_snapshots = _snapshot_foreign_keys_to_summaries(inspector)
    index_snapshots = _snapshot_indexes_on_summaries(inspector)

    with engine.begin() as conn:
        for fk in fk_snapshots:
            if fk["name"]:
                conn.execute(
                    text(f'ALTER TABLE "{fk["table"]}" DROP CONSTRAINT IF EXISTS "{fk["name"]}"')
                )

        conn.execute(text("DROP TABLE IF EXISTS summaries_new CASCADE"))
        conn.execute(CreateTable(summaries_new))

        insert_cols = ", ".join(f'"{n}"' for n in ordered_names)
        select_sql = ", ".join(f'"{n}"' for n in ordered_names)
        conn.execute(
            text(f"INSERT INTO summaries_new ({insert_cols}) SELECT {select_sql} FROM summaries")
        )

        cnt_old = conn.execute(text("SELECT COUNT(*) FROM summaries")).scalar()
        cnt_new = conn.execute(text("SELECT COUNT(*) FROM summaries_new")).scalar()
        if cnt_old != cnt_new:
            raise RuntimeError(f"summaries 重建後列數不一致: old={cnt_old} new={cnt_new}")

        conn.execute(text("DROP TABLE summaries"))
        conn.execute(text("ALTER TABLE summaries_new RENAME TO summaries"))

        for fk in fk_snapshots:
            cols = ", ".join(f'"{c}"' for c in fk["constrained_columns"])
            refs = ", ".join(f'"{c}"' for c in fk["referred_columns"])
            od = fk.get("ondelete")
            extra = f" ON DELETE {od}" if od else ""
            conn.execute(
                text(
                    f'ALTER TABLE "{fk["table"]}" ADD CONSTRAINT "{fk["name"]}" '
                    f"FOREIGN KEY ({cols}) REFERENCES summaries ({refs}){extra}"
                )
            )

        for idx in index_snapshots:
            if not idx["name"] or not idx["column_names"]:
                continue
            cols = ", ".join(f'"{c}"' for c in idx["column_names"])
            un = "UNIQUE " if idx["unique"] else ""
            try:
                conn.execute(
                    text(f'CREATE {un}INDEX IF NOT EXISTS "{idx["name"]}" ON summaries ({cols})')
                )
            except Exception as ex:
                print(f"  ⚠️  還原索引 {idx['name']} 失敗（可稍後手動建立）: {ex}")

        conn.execute(
            text(
                "SELECT setval(pg_get_serial_sequence('summaries','id'), "
                "COALESCE((SELECT MAX(id) FROM summaries), 1))"
            )
        )

    print("  ✓ summaries 已依模型順序重建（資料已保留）")
    return True


def _sync_detection_items(engine):
    from src.database import SessionLocal
    from src.init_detection_items import sync_missing_detection_items

    db = SessionLocal()
    try:
        n = sync_missing_detection_items(db)
        if n:
            print(f"  ✓ 已補齊 {n} 筆 detection_items 預設列（violence / dangerous_items 等）")
        else:
            print("  ✓ detection_items 已含預設項目，無需補齊")
    finally:
        db.close()


def migrate_database():
    """執行資料庫遷移"""
    print("=" * 80)
    print("開始資料庫遷移...")
    print("=" * 80)

    try:
        from src.database import engine, Base, ensure_pgvector_extension
        from src.models import Summary
        from sqlalchemy import text, inspect

        print("\n[0/5] 確保 pgvector 擴展...")
        ensure_pgvector_extension()

        print("\n[1/5] 連接到資料庫...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print("✓ 成功連接到 PostgreSQL")
            print(f"  版本: {version}")

        print("\n[2/5] 檢查現有表結構...")
        inspector = inspect(engine)

        required_tables = ["summaries", "detection_items", "object_crops", "alerts"]
        missing_tables = [t for t in required_tables if not inspector.has_table(t)]

        if missing_tables:
            print(f"⚠️ 缺少以下表: {', '.join(missing_tables)}")
            print("創建所有缺失的表...")
            Base.metadata.create_all(bind=engine)
            print("✓ 所有表已創建")

            if "detection_items" in missing_tables:
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

            if "summaries" in missing_tables:
                # 新建表已由 create_all 依模型順序建立，僅需補齊 detection_items
                print("\n[3/5] 補齊偵測項目字典…")
                _sync_detection_items(engine)
                print("\n" + "=" * 80)
                print("資料庫遷移完成！")
                print("=" * 80)
                return

        print("\n[3/5] 檢查 summaries 欄位…")
        columns = [col["name"] for col in inspector.get_columns("summaries")]
        print(f"  現有欄位數量: {len(columns)}")

        required_columns = {
            "water_flood": "BOOLEAN DEFAULT FALSE",
            "fire": "BOOLEAN DEFAULT FALSE",
            "abnormal_attire_face_cover_at_entry": "BOOLEAN DEFAULT FALSE",
            "person_fallen_unmoving": "BOOLEAN DEFAULT FALSE",
            "double_parking_lane_block": "BOOLEAN DEFAULT FALSE",
            "smoking_outside_zone": "BOOLEAN DEFAULT FALSE",
            "crowd_loitering": "BOOLEAN DEFAULT FALSE",
            "security_door_tamper": "BOOLEAN DEFAULT FALSE",
            "violence": "BOOLEAN DEFAULT FALSE",
            "dangerous_items": "BOOLEAN DEFAULT FALSE",
            "event_reason": "TEXT",
            "events_extra": "TEXT",
            "yolo_detections": "TEXT",
            "yolo_object_count": "TEXT",
            "yolo_crops_dir": "VARCHAR(500)",
            "yolo_total_detections": "INTEGER",
            "yolo_total_frames_processed": "INTEGER",
        }

        missing_columns = []
        for col, col_type in required_columns.items():
            if col not in columns:
                missing_columns.append((col, col_type))
                print(f"  ✗ 缺少欄位: {col} ({col_type})")

        emb_type = _embedding_sql_type(engine)
        if "embedding" not in columns:
            missing_columns.append(("embedding", emb_type))
            print(f"  ✗ 缺少欄位: embedding ({emb_type})")

        if missing_columns:
            print(f"\n  添加 {len(missing_columns)} 個缺少的欄位…")
            with engine.begin() as conn:
                for col_name, col_type in missing_columns:
                    try:
                        sql = f"ALTER TABLE summaries ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
                        print(f"  執行: {sql}")
                        conn.execute(text(sql))
                        print(f"  ✓ 已添加欄位: {col_name}")
                    except Exception as e:
                        print(f"  ✗ 添加欄位 {col_name} 失敗: {e}")
        else:
            print("  ✓ 已知必要欄位皆存在")

        print("\n[4/5] 若欄位順序與模型不一致則重建 summaries…")
        inspector = inspect(engine)
        try:
            _reorder_summaries_table(engine, Summary)
        except Exception as e:
            print(f"  ⚠️ 重建 summaries 欄位順序時發生錯誤（可稍後手動處理）: {e}")
            import traceback

            traceback.print_exc()

        print("\n[5/5] 補齊 detection_items（警報用）…")
        _sync_detection_items(engine)

        inspector = inspect(engine)
        columns_after = [col["name"] for col in inspector.get_columns("summaries")]
        print(f"  目前 summaries 欄位數量: {len(columns_after)}")

        print("\n" + "=" * 80)
        print("資料庫遷移完成！")
        print("=" * 80)
        print("\n建議操作：")
        print("1. 重啟後端容器: docker compose restart backend")
        print("2. 檢查後端日誌: docker logs -f test_platform-main-backend-1")
        print("")

    except Exception as e:
        print(f"\n✗ 遷移失敗: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    migrate_database()
