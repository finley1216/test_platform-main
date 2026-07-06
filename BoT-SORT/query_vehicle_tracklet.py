#!/usr/bin/env python3
"""
Query 圖片 → 重跑 BoT-SORT → 以車輛 CLIP-ReID cosine similarity 比對 tracklet → 輸出拼圖。

功能與 query_tracklet.py 相同，但使用車輛專用 CLIP-ReID 權重與車輛追蹤 JSON，
僅載入 label == "car" 的 crop。

可用鏡頭（--video-ids）：
  車輛追蹤_K8-001, 車輛追蹤_K8-005, 車輛追蹤_K8-007, 車輛追蹤_K8-008,
  車輛追蹤_K8-009, 車輛追蹤_K8-012, 車輛追蹤_K8-014, 車輛追蹤_K8-015,
  車輛追蹤_K8-016, 車輛追蹤_K8-019, 車輛追蹤_K8-020, 車輛追蹤_K8-021,
  車輛追蹤_K8-022, 車輛追蹤_K8-023, 車輛追蹤_K8-028, 車輛追蹤_K8-030

單支鏡頭：
  python3 query_vehicle_tracklet.py \\
    --query-image /mnt/10THDD/M133040024/SSD/ASE/CLIP-ReID-embed-test/data/BSH-5613.jpg \\
    --video-ids 車輛追蹤_K8-009 \\
    --similarity-thresh 0.80

多支鏡頭：
  python3 query_vehicle_tracklet.py \\
    --query-image /mnt/10THDD/M133040024/SSD/ASE/CLIP-ReID-embed-test/data/BSH-5613.jpg \\
    --video-ids 車輛追蹤_K8-001 車輛追蹤_K8-005 車輛追蹤_K8-009 \\
    --similarity-thresh 0.80

調整 BoT-SORT 參數：
  python3 query_vehicle_tracklet.py \\
    --query-image /mnt/10THDD/M133040024/SSD/ASE/CLIP-ReID-embed-test/data/BSH-5613.jpg \\
    --video-ids 車輛追蹤_K8-009 \\
    --similarity-thresh 0.80 \\
    --appearance-thresh 0.50 \\
    --proximity-thresh 0.80 \\
    --match-thresh 0.75 \\
    --track-buffer 20 \\
    --new-track-thresh 0.65
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CLIP-ReID"))
sys.path.insert(0, str(HERE))

from repo_paths import CLIP_REID_ROOT, OUTPUT_ROOT  # noqa: E402

# 必須先安裝 stub 才能 import 原始腳本
import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

from run_k809_botsort_clipreid import (  # noqa: E402
    CLIP_REID_ROOT,
    CROP_SEG_RE,
    compute_global_frame,
    collect_track_rows,
    extract_clipreid_embeddings,
    group_frames,
    init_botsort,
    run_botsort_tracking,
)
from clipreid_crossmodal import ClipReIDEmbedder, add_common_model_args  # noqa: E402

from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

DEFAULT_MAPPING = OUTPUT_ROOT / "車輛追蹤_crop_time_mapping.json"
DEFAULT_VIDEO_IDS = ["車輛追蹤_K8-009"]
DEFAULT_OUTPUT_DIR = HERE / "output" / "query_results"

# 拼圖常數（與 tracking_collage.py 一致）
TILE_W = 120
TILE_H = 200
CAPTION_H = 40
CARD_PAD = 4
HEADER_H = 24
ROW_SEP = 3
ROW_GAP = 10
REP_GAP = 28
TILE_GAP = 8
OUTER_PAD = 16
MIN_CANVAS_W = 2000
BG = (245, 245, 245)
REP_OUTLINE = (0, 160, 0)
TILE_OUTLINE = (180, 180, 180)


def vehicle_crop_dir(video_id: str) -> Path:
    return OUTPUT_ROOT / video_id


def vehicle_output_dir(video_id: str) -> Path:
    slug = video_id.lower().replace("-", "")
    return OUTPUT_ROOT / "embed_cache" / slug


def vehicle_cache_path(video_id: str) -> Path:
    return vehicle_output_dir(video_id) / "vehicle_clipreid_embeddings_cache.pkl"


def load_vehicle_records(mapping_json: Path, crop_dir: Path, video_id: str) -> List[Dict[str, Any]]:
    print(f"[1] 讀取 JSON: {mapping_json}")
    with mapping_json.open(encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = []
    for seg in data.get("segments", []):
        if seg.get("video_id") != video_id:
            continue
        for crop in seg.get("crops", []):
            if crop.get("label") != "car":
                continue
            name = crop["crop_path"]
            m = CROP_SEG_RE.search(name)
            seg_i = int(m.group(1)) if m else 0
            frame = int(crop["frame"])
            records.append(
                {
                    "frame": frame,
                    "global_frame": compute_global_frame(seg_i, frame),
                    "segment_idx": seg_i,
                    "box": json.loads(crop["box"]),
                    "score": float(crop["score"]),
                    "crop_path": str(crop_dir / name),
                    "relative_seconds": float(crop.get("relative_seconds", 0.0)),
                    "absolute_timestamp": str(crop.get("absolute_timestamp", "")),
                }
            )
    records.sort(key=lambda r: (r["global_frame"], -r["score"]))
    print(f"    video_id={video_id} 共 {len(records)} 筆 car crop")
    return records


def _font(size: int = 12):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_tile(path: str, tw: int = TILE_W, th: int = TILE_H) -> Image.Image:
    p = Path(path)
    if not p.is_file():
        alt = Path(str(path).replace("/home/M133040024/ASE/", "/mnt/10THDD/M133040024/SSD/ASE/"))
        if alt.is_file():
            p = alt
    try:
        return ImageOps.pad(Image.open(p).convert("RGB"), (tw, th), color=(0, 0, 0))
    except Exception:
        img = Image.new("RGB", (tw, th), color=(30, 30, 30))
        ImageDraw.Draw(img).text((6, 6), "load fail", fill=(220, 60, 60))
        return img


def _card_size() -> Tuple[int, int]:
    return TILE_W + CARD_PAD * 2, TILE_H + CAPTION_H + CARD_PAD * 2


def _draw_tile_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    rec: Dict[str, Any],
    outline: Tuple[int, int, int],
    width: int,
    font,
) -> None:
    cw, ch = _card_size()
    draw.rectangle([(x, y), (x + cw - 1, y + ch - 1)], outline=outline, width=width)
    ix = x + CARD_PAD
    iy = y + CARD_PAD
    canvas.paste(_load_tile(rec["crop_path"]), (ix, iy))
    name = Path(rec["crop_path"]).name
    line1 = name if len(name) <= 28 else "..." + name[-25:]
    line2 = f"f={rec['global_frame']}  s={rec['score']:.2f}"
    ty = iy + TILE_H + 4
    draw.text((ix, ty), line1, fill=(0, 0, 0), font=font)
    draw.text((ix, ty + 14), line2, fill=(60, 60, 60), font=font)


def _measure_tiles_block(rest: List[Dict[str, Any]], tiles_area_w: int) -> int:
    if not rest:
        return _card_size()[1]
    cw, ch = _card_size()
    step = cw + TILE_GAP
    per_row = max(1, (tiles_area_w + TILE_GAP) // step)
    n_rows = (len(rest) + per_row - 1) // per_row
    return n_rows * ch + (n_rows - 1) * TILE_GAP


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float64)
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-6)


def build_tracklet_vec(crops: List[Dict[str, Any]], emb_cache: Dict[str, np.ndarray]) -> np.ndarray:
    embs = np.stack([emb_cache[c["crop_path"]] for c in crops], axis=0).astype(np.float64)
    return normalize_vec(embs.mean(axis=0))


def save_query_collage(
    *,
    matched: List[Dict[str, Any]],
    out_path: Path,
    canvas_w: int = MIN_CANVAS_W,
) -> None:
    font = _font(11)
    header_font = _font(12)
    card_w, card_h = _card_size()

    min_row_w = OUTER_PAD + card_w + REP_GAP + card_w + OUTER_PAD
    canvas_w = max(canvas_w, MIN_CANVAS_W, min_row_w)
    tiles_area_w = canvas_w - 2 * OUTER_PAD - card_w - REP_GAP
    right_limit = canvas_w - OUTER_PAD

    blocks: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], int]] = []
    total_h = OUTER_PAD
    for entry in matched:
        crops = entry["crops"]
        crops_sorted = sorted(crops, key=lambda r: (r["global_frame"], -r["score"]))
        rep = max(crops_sorted, key=lambda r: r["score"])
        rest = [c for c in crops_sorted if c is not rep]
        t0 = entry["start_time"]
        t1 = entry["end_time"]
        header = (
            f"track_id={entry['track_id']}  sim={entry['similarity']:.3f}  "
            f"n={entry['n_crops']}  {t0} --> {t1}"
        )
        tiles_h = _measure_tiles_block(rest, tiles_area_w)
        cards_h = max(card_h, tiles_h)
        block_h = HEADER_H + ROW_GAP + cards_h
        blocks.append((header, rep, rest, block_h))
        total_h += block_h + ROW_SEP + ROW_GAP

    canvas_h = total_h + OUTER_PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=BG)
    draw = ImageDraw.Draw(canvas)
    y = OUTER_PAD

    for header, rep, rest, block_h in blocks:
        draw.rectangle(
            [(OUTER_PAD, y), (canvas_w - OUTER_PAD, y + HEADER_H - 1)],
            fill=(255, 255, 255),
            outline=(200, 200, 200),
        )
        draw.text((OUTER_PAD + 6, y + 5), header, fill=(0, 0, 0), font=header_font)
        y_cards = y + HEADER_H + ROW_GAP

        _draw_tile_card(
            canvas, draw, x=OUTER_PAD, y=y_cards, rec=rep,
            outline=REP_OUTLINE, width=3, font=font,
        )

        x0 = OUTER_PAD + card_w + REP_GAP
        cx, cy = x0, y_cards
        for rec in rest:
            if cx + card_w > right_limit and cx > x0:
                cx = x0
                cy += card_h + TILE_GAP
            _draw_tile_card(
                canvas, draw, x=cx, y=cy, rec=rec,
                outline=TILE_OUTLINE, width=1, font=font,
            )
            cx += card_w + TILE_GAP

        y += block_h + ROW_SEP + ROW_GAP
        sep_y = y - ROW_GAP - ROW_SEP // 2
        draw.line(
            [(OUTER_PAD, sep_y), (canvas_w - OUTER_PAD, sep_y)],
            fill=(210, 210, 210),
            width=ROW_SEP,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"[OK] 拼圖 -> {out_path} ({canvas_w}x{canvas_h}, {len(matched)} tracks)")


def process_video(
    *,
    video_id: str,
    query_vec: np.ndarray,
    embedder: ClipReIDEmbedder,
    mapping_json: Path,
    output_dir: Path,
    similarity_thresh: float,
    query_image: Path,
    botsort_kwargs: Dict[str, Any],
) -> None:
    print(f"\n========== [{video_id}] ==========")
    crop_dir = vehicle_crop_dir(video_id)
    cache_path = vehicle_cache_path(video_id)

    print(f"[{video_id}][1] 載入 records …")
    records = load_vehicle_records(mapping_json, crop_dir, video_id=video_id)
    if not records:
        print(f"[{video_id}] 無 car crop 記錄，跳過")
        return

    print(f"[{video_id}][2] 載入 / 抽取 embedding（快取 {cache_path}）…")
    emb_cache = extract_clipreid_embeddings(records, embedder, cache_path)
    frame_groups = group_frames(records)

    print(f"[{video_id}][3] 重跑 BoT-SORT …")
    tracker = init_botsort(video_id=video_id, **botsort_kwargs)
    rows = run_botsort_tracking(
        frame_groups,
        emb_cache,
        tracker,
        video_id=video_id,
        reset_per_segment=False,
    )
    track_rows = collect_track_rows(rows)
    print(f"[{video_id}]    共 {len(track_rows)} 條 tracklet")

    print(f"[{video_id}][4] 建立 tracklet 代表向量並比對 query …")
    candidates: List[Dict[str, Any]] = []
    for track_id, crops in track_rows:
        tracklet_vec = build_tracklet_vec(crops, emb_cache)
        similarity = float(np.dot(query_vec, tracklet_vec))
        crops_sorted = sorted(crops, key=lambda r: (r["global_frame"], -r["score"]))
        candidates.append(
            {
                "track_id": track_id,
                "similarity": similarity,
                "n_crops": len(crops),
                "start_time": crops_sorted[0]["absolute_timestamp"],
                "end_time": crops_sorted[-1]["absolute_timestamp"],
                "crops": crops_sorted,
                "crop_paths": [c["crop_path"] for c in crops_sorted],
            }
        )

    matched = [c for c in candidates if c["similarity"] >= similarity_thresh]
    matched.sort(key=lambda x: -x["similarity"])

    print(
        f"[{video_id}] 共 {len(track_rows)} 條 tracklet，"
        f"找到 {len(matched)} 條相似度 >= {similarity_thresh:.2f}"
    )
    for entry in matched:
        print(
            f"  track_id={entry['track_id']}  similarity={entry['similarity']:.3f}  "
            f"n={entry['n_crops']}  {entry['start_time']} ~ {entry['end_time']}"
        )

    if not matched:
        print(f"[{video_id}] 找不到相似 tracklet >= {similarity_thresh:.2f}，仍輸出空 JSON")
    else:
        print(f"[{video_id}][5] 輸出拼圖 …")
        collage_path = output_dir / f"{video_id}_vehicle_query_result.png"
        save_query_collage(matched=matched, out_path=collage_path)

    print(f"[{video_id}][6] 儲存 JSON …")
    json_path = output_dir / f"{video_id}_vehicle_query_result.json"
    json_data = {
        "query_image": str(query_image.resolve()),
        "video_id": video_id,
        "similarity_thresh": similarity_thresh,
        "matched_tracks": [
            {
                "track_id": e["track_id"],
                "similarity": round(e["similarity"], 6),
                "n_crops": e["n_crops"],
                "start_time": e["start_time"],
                "end_time": e["end_time"],
                "crop_paths": e["crop_paths"],
            }
            for e in matched
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON -> {json_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query 圖片比對車輛 BoT-SORT tracklet")
    p.add_argument("--query-image", type=Path, required=True)
    p.add_argument("--video-ids", nargs="+", default=DEFAULT_VIDEO_IDS)
    p.add_argument("--similarity-thresh", type=float, default=0.80)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--appearance-thresh", type=float, default=0.50)
    p.add_argument("--proximity-thresh", type=float, default=0.80)
    p.add_argument("--match-thresh", type=float, default=0.75)
    p.add_argument("--track-buffer", type=int, default=20)
    p.add_argument("--new-track-thresh", type=float, default=0.65)
    p.add_argument("--track-high-thresh", type=float, default=0.35)
    p.add_argument("--track-low-thresh", type=float, default=0.10)
    add_common_model_args(p)
    p.set_defaults(
        config_file=str(CLIP_REID_ROOT / "configs" / "veri" / "vit_prom.yml"),
        weight=str(CLIP_REID_ROOT / "pretrained" / "ViT_CLIP_ReID_SIE_OLP_VeRi.pth"),
        num_classes=576,
        camera_num=20,
        view_num=8,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(HERE)

    query_image = args.query_image.resolve()
    if not query_image.is_file():
        raise SystemExit(f"query 圖片不存在：{query_image}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    botsort_kwargs = {
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "new_track_thresh": args.new_track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "proximity_thresh": args.proximity_thresh,
        "appearance_thresh": args.appearance_thresh,
    }

    print("[0] 初始化 ClipReIDEmbedder（車輛）…")
    embedder = ClipReIDEmbedder(
        repo_root=args.repo_root,
        config_file=args.config_file,
        weight=args.weight,
        num_classes=args.num_classes,
        camera_num=args.camera_num,
        view_num=args.view_num,
    )
    print("    CLIP-ReID 就緒")

    print(f"[1] 抽取 query embedding：{query_image}")
    query_vec = normalize_vec(embedder.encode_one(str(query_image)))
    print(f"    query_vec shape=({query_vec.shape[0]},)")

    for video_id in args.video_ids:
        process_video(
            video_id=video_id,
            query_vec=query_vec,
            embedder=embedder,
            mapping_json=args.mapping_json,
            output_dir=output_dir,
            similarity_thresh=args.similarity_thresh,
            query_image=query_image,
            botsort_kwargs=botsort_kwargs,
        )

    print(f"\n[完成] 輸出目錄：{output_dir}")


if __name__ == "__main__":
    main()
