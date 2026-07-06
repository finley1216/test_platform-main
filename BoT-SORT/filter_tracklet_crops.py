#!/usr/bin/env python3
"""
對 query_result.json 中某條 tracklet 做 CLIP-ReID + 時空約束 intra-tracklet 過濾。

參考集：對 query 相似度最高的 Top-K 張（非清單前 N 張）。

score-mode independent（預設）：三個獨立門檻 sim / bbox / time
score-mode combined：combined = alpha*sim_to_query + (1-alpha)*mean_sim_to_ref

independent 模式（原本）：
  python3 BoT-SORT-K809/filter_tracklet_crops.py \\
    --query-result-json BoT-SORT-K809/output/query_results/人員追蹤_20260528_K8-09_query_result.json \\
    --query-image CLIP-ReID-embed-test/data/p9.jpg \\
    --track-id 24 \\
    --score-mode independent \\
    --sim-thresh 0.75 --bbox-thresh 300 --time-thresh 60

combined 模式：
  python3 BoT-SORT-K809/filter_tracklet_crops.py \\
    --query-result-json BoT-SORT-K809/output/query_results/人員追蹤_20260528_K8-09_query_result.json \\
    --query-image CLIP-ReID-embed-test/data/p9.jpg \\
    --track-id 24 \\
    --score-mode combined --alpha 0.5 --combined-thresh 0.82

combined + tune：
  python3 BoT-SORT-K809/filter_tracklet_crops.py ... --track-id 24 --score-mode combined --tune-mode

輸出（所有 track／模式／是否 tune 皆相同）：
  output/filter_results/track_{track_id}_filter_result.png
  output/filter_results/track_{track_id}_filter_result.json
  tune 時 JSON 含 tune_mode=true 與 tune_metrics；參數為搜尋最佳值。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CLIP-ReID"))
sys.path.insert(0, str(HERE))

from repo_paths import CLIP_REID_ROOT, OUTPUT_ROOT  # noqa: E402

CLIP_EMBED_ROOT = CLIP_REID_ROOT
ASE_ROOT = WORKSPACE_ROOT

from clipreid_crossmodal import ClipReIDEmbedder, add_common_model_args  # noqa: E402

DEFAULT_MAPPING = OUTPUT_ROOT / "人員追蹤_crop_time_mapping.json"
FALLBACK_MAPPING_20260528 = OUTPUT_ROOT / "人員追蹤_20260528_crop_time_mapping.json"
DEFAULT_CACHE = HERE / "output" / "K8-09" / "clipreid_embeddings_cache.pkl"
DEFAULT_OUTPUT = OUTPUT_ROOT / "filter_results"

GT_REJECTED_TRACK24 = {
    "crop_s003_5_2_person.jpg",
    "crop_s003_10_2_person.jpg",
    "crop_s003_16_2_person.jpg",
}

TILE_W = 120
TILE_H = 200
CAPTION_H = 56
CARD_PAD = 4
HEADER_H = 32
ROW_GAP = 12
TILE_GAP = 8
OUTER_PAD = 16
CANVAS_W = 2000
BG = (245, 245, 245)
KEPT_HEADER_BG = (220, 255, 220)
REJECT_HEADER_BG = (255, 220, 220)


@dataclass
class CropRecord:
    index: int
    crop_name: str
    image_path: str
    box: Tuple[float, float, float, float]
    timestamp: datetime
    timestamp_str: str
    embedding: Optional[np.ndarray] = None
    sim_to_query: float = 0.0
    sim_to_ref: float = 0.0
    bbox_dist: float = 0.0
    time_gap: float = 0.0
    is_ref: bool = False
    kept: bool = True
    reject_reasons: List[str] = field(default_factory=list)
    combined_score: float = 0.0


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_file():
        return path.resolve()
    s = str(p)
    for old, new in (
        ("/home/M133040024/ASE/", str(ASE_ROOT) + "/"),
        ("/mnt/10THDD/M133040024/SSD/ASE/", str(ASE_ROOT) + "/"),
    ):
        cand = Path(s.replace(old, new))
        if cand.is_file():
            return cand.resolve()
    return path


def parse_box(box_raw) -> Tuple[float, float, float, float]:
    if isinstance(box_raw, str):
        vals = json.loads(box_raw)
    elif isinstance(box_raw, (list, tuple)):
        vals = box_raw
    else:
        raise ValueError(f"invalid box: {box_raw}")
    x1, y1, x2, y2 = [float(v) for v in vals]
    return x1, y1, x2, y2


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a.astype(np.float64)), l2_normalize(b.astype(np.float64))))


def _merge_mapping_segments(
    lookup: Dict[str, dict],
    payload: dict,
    video_ids: Sequence[str],
) -> None:
    vid_set = set(video_ids)
    for seg in payload.get("segments", []):
        vid = str(seg.get("video_id", ""))
        if vid_set and vid not in vid_set and not any(v in vid for v in vid_set):
            continue
        for crop in seg.get("crops", []):
            name = Path(str(crop.get("crop_path", ""))).name
            if not name:
                continue
            ts = crop.get("absolute_timestamp")
            if not ts:
                continue
            lookup[name] = {
                "box": parse_box(crop.get("box")),
                "absolute_timestamp": str(ts),
                "video_id": vid,
            }


def load_bbox_lookup(
    mapping_json: Path,
    video_ids: Sequence[str],
    *,
    fallback_json: Optional[Path] = None,
) -> Dict[str, dict]:
    """crop 檔名 -> {box, absolute_timestamp}；fallback 只補 primary 缺少的檔名。"""
    lookup: Dict[str, dict] = {}
    _merge_mapping_segments(lookup, json.loads(mapping_json.read_text(encoding="utf-8")), video_ids)
    if fallback_json and fallback_json.is_file():
        before = len(lookup)
        fb_lookup: Dict[str, dict] = {}
        _merge_mapping_segments(
            fb_lookup,
            json.loads(fallback_json.read_text(encoding="utf-8")),
            video_ids,
        )
        for name, meta in fb_lookup.items():
            if name not in lookup:
                lookup[name] = meta
        added = len(lookup) - before
        if added:
            print(f"    fallback 補齊 {added} 筆: {fallback_json.name}")
    return lookup


def find_track(query_payload: dict, track_id: int) -> dict:
    for t in query_payload.get("matched_tracks", []):
        if int(t.get("track_id")) == track_id:
            return t
    raise ValueError(f"找不到 track_id={track_id}")


def build_records(
    track: dict,
    bbox_lookup: Dict[str, dict],
) -> List[CropRecord]:
    records: List[CropRecord] = []
    for i, raw in enumerate(track.get("crop_paths", [])):
        resolved = resolve_path(str(raw))
        if not resolved.is_file():
            raise FileNotFoundError(f"crop 不存在: {raw}")
        name = resolved.name
        meta = bbox_lookup.get(name)
        if meta is None:
            raise KeyError(f"mapping 找不到 bbox/時間: {name}")
        ts = parse_ts(meta["absolute_timestamp"])
        records.append(
            CropRecord(
                index=i,
                crop_name=name,
                image_path=str(resolved),
                box=meta["box"],
                timestamp=ts,
                timestamp_str=meta["absolute_timestamp"],
            )
        )
    return records


def load_embedding_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    if not cache_path.is_file():
        print(f"    [Info] 快取不存在，將新建: {cache_path}")
        return {}
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    print(f"    載入快取 {len(cache)} 筆")
    return cache


def cache_lookup(cache: Dict[str, np.ndarray], image_path: str) -> Optional[np.ndarray]:
    p = resolve_path(image_path)
    keys = [
        str(p),
        image_path,
        str(Path(image_path)),
    ]
    for k in keys:
        if k in cache:
            return cache[k]
    # 以檔名比對（路徑前綴可能不同）
    name = p.name
    for k, v in cache.items():
        if Path(k).name == name:
            return v
    return None


def ensure_embeddings(
    records: List[CropRecord],
    cache: Dict[str, np.ndarray],
    cache_path: Path,
    embedder: ClipReIDEmbedder,
    query_path: Path,
) -> np.ndarray:
    print("[3] 載入 / 補齊 crop embedding …")
    missing: List[CropRecord] = []
    for rec in records:
        emb = cache_lookup(cache, rec.image_path)
        if emb is None:
            missing.append(rec)
        else:
            rec.embedding = emb.astype(np.float32)

    if missing:
        print(f"    快取缺少 {len(missing)} 張，CLIP-ReID 補抽 …")
        paths = [r.image_path for r in missing]
        feats = embedder.encode_images(paths, batch_size=64)
        for rec, feat in zip(missing, feats):
            rec.embedding = feat.astype(np.float32)
            cache[rec.image_path] = rec.embedding
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(cache, f)
        print(f"    快取已更新: {cache_path}")
    else:
        print("    全部 crop 命中快取")

    print(f"[4] query embedding: {query_path}")
    q_emb = embedder.encode_one(str(query_path)).astype(np.float32)
    return l2_normalize(q_emb)


def select_reference_set(
    records: List[CropRecord],
    query_emb: np.ndarray,
    top_k: int,
) -> List[int]:
    print(f"[5] 選參考集 Top-{top_k}（對 query 相似度最高）…")
    for rec in records:
        assert rec.embedding is not None
        rec.sim_to_query = cosine_sim(rec.embedding, query_emb)

    ranked = sorted(range(len(records)), key=lambda i: -records[i].sim_to_query)
    ref_indices = ranked[: min(top_k, len(records))]
    for i in ref_indices:
        records[i].is_ref = True
        records[i].kept = True

    for rank, i in enumerate(ref_indices, 1):
        r = records[i]
        print(f"    ref#{rank}  [{i}] {r.crop_name}  sim_to_query={r.sim_to_query:.4f}")
    return ref_indices


def compute_metrics(
    records: List[CropRecord],
    ref_indices: List[int],
) -> None:
    refs = [records[i] for i in ref_indices]
    ref_centers = np.array([box_center(r.box) for r in refs], dtype=np.float64)
    mean_center = ref_centers.mean(axis=0)
    ref_embs = [r.embedding for r in refs]
    assert all(e is not None for e in ref_embs)

    for rec in records:
        if rec.is_ref:
            rec.sim_to_ref = 1.0
            rec.bbox_dist = 0.0
            rec.time_gap = 0.0
            continue
        assert rec.embedding is not None
        rec.sim_to_ref = float(
            np.mean([cosine_sim(rec.embedding, e) for e in ref_embs])
        )
        cx, cy = box_center(rec.box)
        rec.bbox_dist = float(
            np.linalg.norm(np.array([cx, cy]) - mean_center)
        )
        rec.time_gap = min(
            abs((rec.timestamp - r.timestamp).total_seconds()) for r in refs
        )


def apply_filter(
    records: List[CropRecord],
    *,
    sim_thresh: float,
    bbox_thresh: float,
    time_thresh: float,
    verbose: bool = True,
) -> Tuple[List[CropRecord], List[CropRecord]]:
    if verbose:
        print(
            f"[6] 過濾  sim>={sim_thresh}  bbox<={bbox_thresh}px  "
            f"time<={time_thresh}s"
        )
    kept: List[CropRecord] = []
    rejected: List[CropRecord] = []

    for rec in records:
        if rec.is_ref:
            kept.append(rec)
            continue
        reasons: List[str] = []
        if rec.sim_to_ref < sim_thresh:
            reasons.append("sim")
        if rec.bbox_dist > bbox_thresh:
            reasons.append("bbox")
        if rec.time_gap > time_thresh:
            reasons.append("time")
        rec.reject_reasons = reasons
        rec.kept = len(reasons) == 0
        if verbose:
            status = "KEPT" if rec.kept else "REJECTED"
            reason_tag = f" ({'+'.join(reasons)})" if reasons else ""
            print(
                f"    {rec.crop_name:35s}  sim={rec.sim_to_ref:.2f}  "
                f"bbox={rec.bbox_dist:.0f}px  time={rec.time_gap:.0f}s  "
                f"→ {status}{reason_tag}"
            )
        (kept if rec.kept else rejected).append(rec)

    kept.sort(key=lambda r: r.timestamp)
    rejected.sort(key=lambda r: r.timestamp)
    if verbose:
        print(f"    結果: kept={len(kept)}  rejected={len(rejected)}")
    return kept, rejected


def compute_mean_sim_to_ref(
    records: List[CropRecord],
    ref_indices: List[int],
) -> None:
    """combined 模式：計算 mean_sim_to_ref（寫入 rec.sim_to_ref）。"""
    print("    計算 mean_sim_to_ref（對參考集 Top-K 平均 cosine）…")
    refs = [records[i] for i in ref_indices]
    ref_embs = [r.embedding for r in refs]
    assert all(e is not None for e in ref_embs)
    for rec in records:
        if rec.is_ref:
            rec.sim_to_ref = 1.0
            continue
        assert rec.embedding is not None
        rec.sim_to_ref = float(
            np.mean([cosine_sim(rec.embedding, e) for e in ref_embs])
        )


def compute_combined_scores(
    records: List[CropRecord],
    alpha: float,
) -> None:
    print(f"[6] combined 評分  alpha={alpha}  (1-alpha)={1-alpha:.2f}")
    for rec in records:
        if rec.is_ref:
            rec.combined_score = 1.0
        else:
            rec.combined_score = float(
                alpha * rec.sim_to_query + (1.0 - alpha) * rec.sim_to_ref
            )


def apply_filter_combined(
    records: List[CropRecord],
    *,
    combined_thresh: float,
    verbose: bool = True,
) -> Tuple[List[CropRecord], List[CropRecord]]:
    if verbose:
        print(f"[7] combined 過濾  thresh={combined_thresh}  (低於此剔除)")
    kept: List[CropRecord] = []
    rejected: List[CropRecord] = []

    for rec in records:
        if rec.is_ref:
            kept.append(rec)
            if verbose:
                print(
                    f"  {rec.crop_name}\n"
                    f"    sim_to_query={rec.sim_to_query:.3f}  "
                    f"mean_sim_ref={rec.sim_to_ref:.3f}  "
                    f"combined={rec.combined_score:.3f} → KEPT (ref)"
                )
            continue
        rec.kept = rec.combined_score >= combined_thresh
        if not rec.kept:
            rec.reject_reasons = ["combined"]
        status = "KEPT" if rec.kept else "REJECTED"
        if verbose:
            print(
                f"  {rec.crop_name}\n"
                f"    sim_to_query={rec.sim_to_query:.3f}  "
                f"mean_sim_ref={rec.sim_to_ref:.3f}  "
                f"combined={rec.combined_score:.3f} → {status}"
            )
        (kept if rec.kept else rejected).append(rec)

    kept.sort(key=lambda r: r.timestamp)
    rejected.sort(key=lambda r: r.timestamp)
    if verbose:
        print(f"    結果: kept={len(kept)}  rejected={len(rejected)}")
    return kept, rejected


def _font(size: int = 11):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _card_size() -> Tuple[int, int]:
    return TILE_W + CARD_PAD * 2, TILE_H + CAPTION_H + CARD_PAD * 2


def _load_tile(path: str) -> Image.Image:
    try:
        return ImageOps.pad(
            Image.open(resolve_path(path)).convert("RGB"),
            (TILE_W, TILE_H),
            color=(0, 0, 0),
        )
    except Exception:
        img = Image.new("RGB", (TILE_W, TILE_H), color=(40, 40, 40))
        ImageDraw.Draw(img).text((8, 8), "fail", fill=(220, 60, 60))
        return img


def _draw_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    title: str,
    header_bg: Tuple[int, int, int],
    crops: List[CropRecord],
    outline: Tuple[int, int, int],
) -> int:
    font = _font(10)
    header_font = _font(12)
    card_w, card_h = _card_size()
    cols = max(1, (CANVAS_W - 2 * OUTER_PAD + TILE_GAP) // (card_w + TILE_GAP))
    n_rows = max(1, (len(crops) + cols - 1) // cols) if crops else 1
    block_h = HEADER_H + ROW_GAP + n_rows * card_h + (n_rows - 1) * TILE_GAP

    draw.rectangle(
        [(OUTER_PAD, y), (CANVAS_W - OUTER_PAD, y + HEADER_H - 1)],
        fill=header_bg,
        outline=(200, 200, 200),
    )
    draw.text((OUTER_PAD + 8, y + 8), title, fill=(0, 0, 0), font=header_font)

    y_cards = y + HEADER_H + ROW_GAP
    for rank, rec in enumerate(crops):
        row = rank // cols
        col = rank % cols
        x = OUTER_PAD + col * (card_w + TILE_GAP)
        cy = y_cards + row * (card_h + TILE_GAP)
        draw.rectangle(
            [(x, cy), (x + card_w - 1, cy + card_h - 1)],
            outline=outline,
            width=2 if rank == 0 else 1,
        )
        ix, iy = x + CARD_PAD, cy + CARD_PAD
        canvas.paste(_load_tile(rec.image_path), (ix, iy))
        line1 = rec.crop_name if len(rec.crop_name) <= 24 else "..." + rec.crop_name[-21:]
        line2 = f"sim={rec.sim_to_ref:.2f}  bbox={rec.bbox_dist:.0f}px"
        line3 = f"time={rec.time_gap:.0f}s"
        if rec.reject_reasons:
            line3 += f"  [{ '+'.join(rec.reject_reasons)}]"
        ty = iy + TILE_H + 3
        draw.text((ix, ty), line1, fill=(0, 0, 0), font=font)
        draw.text((ix, ty + 13), line2, fill=(60, 60, 60), font=font)
        draw.text((ix, ty + 26), line3, fill=(120, 40, 40) if rec.reject_reasons else (80, 80, 80), font=font)

    if not crops:
        draw.text((OUTER_PAD + 8, y_cards + 20), "(empty)", fill=(120, 120, 120), font=font)

    return block_h


def save_filter_collage(
    *,
    kept: List[CropRecord],
    rejected: List[CropRecord],
    track_id: int,
    sim_thresh: float,
    bbox_thresh: float,
    time_thresh: float,
    out_path: Path,
) -> None:
    print(f"[7] 輸出拼圖 -> {out_path}")
    kept_title = (
        f"KEPT  n={len(kept)}  track_id={track_id}  "
        f"sim_thresh={sim_thresh}  bbox_thresh={bbox_thresh}  time_thresh={time_thresh}"
    )
    rej_title = f"REJECTED  n={len(rejected)}"

    # 先量高度
    card_w, card_h = _card_size()
    cols = max(1, (CANVAS_W - 2 * OUTER_PAD + TILE_GAP) // (card_w + TILE_GAP))

    def row_h(n: int) -> int:
        nr = max(1, (n + cols - 1) // cols) if n else 1
        return HEADER_H + ROW_GAP + nr * card_h + (nr - 1) * TILE_GAP

    canvas_h = OUTER_PAD + row_h(len(kept)) + ROW_GAP + row_h(len(rejected)) + OUTER_PAD
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), color=BG)
    draw = ImageDraw.Draw(canvas)
    y = OUTER_PAD
    y += _draw_row(
        canvas,
        draw,
        y=y,
        title=kept_title,
        header_bg=KEPT_HEADER_BG,
        crops=kept,
        outline=(0, 160, 0),
    )
    y += ROW_GAP
    _draw_row(
        canvas,
        draw,
        y=y,
        title=rej_title,
        header_bg=REJECT_HEADER_BG,
        crops=rejected,
        outline=(200, 50, 50),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"    完成 {CANVAS_W}x{canvas_h}")


def _draw_row_combined(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    title: str,
    header_bg: Tuple[int, int, int],
    crops: List[CropRecord],
    outline: Tuple[int, int, int],
    show_rejected_mark: bool,
) -> int:
    font = _font(10)
    header_font = _font(12)
    card_w, card_h = _card_size()
    cols = max(1, (CANVAS_W - 2 * OUTER_PAD + TILE_GAP) // (card_w + TILE_GAP))
    n_rows = max(1, (len(crops) + cols - 1) // cols) if crops else 1
    block_h = HEADER_H + ROW_GAP + n_rows * card_h + (n_rows - 1) * TILE_GAP

    draw.rectangle(
        [(OUTER_PAD, y), (CANVAS_W - OUTER_PAD, y + HEADER_H - 1)],
        fill=header_bg,
        outline=(200, 200, 200),
    )
    draw.text((OUTER_PAD + 8, y + 8), title, fill=(0, 0, 0), font=header_font)

    y_cards = y + HEADER_H + ROW_GAP
    for rank, rec in enumerate(crops):
        row = rank // cols
        col = rank % cols
        x = OUTER_PAD + col * (card_w + TILE_GAP)
        cy = y_cards + row * (card_h + TILE_GAP)
        draw.rectangle(
            [(x, cy), (x + card_w - 1, cy + card_h - 1)],
            outline=outline,
            width=2 if rank == 0 else 1,
        )
        ix, iy = x + CARD_PAD, cy + CARD_PAD
        canvas.paste(_load_tile(rec.image_path), (ix, iy))
        line1 = rec.crop_name if len(rec.crop_name) <= 24 else "..." + rec.crop_name[-21:]
        line2 = f"q={rec.sim_to_query:.2f}  ref={rec.sim_to_ref:.2f}"
        line3 = f"combined={rec.combined_score:.3f}"
        if show_rejected_mark and rec.reject_reasons:
            line3 += "  ✗"
        ty = iy + TILE_H + 3
        draw.text((ix, ty), line1, fill=(0, 0, 0), font=font)
        draw.text((ix, ty + 13), line2, fill=(60, 60, 60), font=font)
        draw.text(
            (ix, ty + 26),
            line3,
            fill=(180, 40, 40) if show_rejected_mark else (80, 80, 80),
            font=font,
        )

    if not crops:
        draw.text((OUTER_PAD + 8, y_cards + 20), "(empty)", fill=(120, 120, 120), font=font)

    return block_h


def save_filter_collage_combined(
    *,
    kept: List[CropRecord],
    rejected: List[CropRecord],
    track_id: int,
    alpha: float,
    combined_thresh: float,
    out_path: Path,
) -> None:
    print(f"[8] 輸出拼圖 (combined) -> {out_path}")
    kept_title = (
        f"KEPT  n={len(kept)}  track_id={track_id}  "
        f"mode=combined  alpha={alpha}  thresh={combined_thresh}"
    )
    rej_title = f"REJECTED  n={len(rejected)}"

    card_w, card_h = _card_size()
    cols = max(1, (CANVAS_W - 2 * OUTER_PAD + TILE_GAP) // (card_w + TILE_GAP))

    def row_h(n: int) -> int:
        nr = max(1, (n + cols - 1) // cols) if n else 1
        return HEADER_H + ROW_GAP + nr * card_h + (nr - 1) * TILE_GAP

    canvas_h = OUTER_PAD + row_h(len(kept)) + ROW_GAP + row_h(len(rejected)) + OUTER_PAD
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), color=BG)
    draw = ImageDraw.Draw(canvas)
    y = OUTER_PAD
    y += _draw_row_combined(
        canvas,
        draw,
        y=y,
        title=kept_title,
        header_bg=KEPT_HEADER_BG,
        crops=kept,
        outline=(0, 160, 0),
        show_rejected_mark=False,
    )
    y += ROW_GAP
    _draw_row_combined(
        canvas,
        draw,
        y=y,
        title=rej_title,
        header_bg=REJECT_HEADER_BG,
        crops=rejected,
        outline=(200, 50, 50),
        show_rejected_mark=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"    完成 {CANVAS_W}x{canvas_h}")


def filter_result_paths(output_dir: Path, track_id: int) -> Tuple[Path, Path]:
    """所有模式／tune 皆使用相同檔名。"""
    d = output_dir.resolve()
    return (
        d / f"track_{track_id}_filter_result.png",
        d / f"track_{track_id}_filter_result.json",
    )


def write_independent_result_json(
    json_path: Path,
    *,
    track_id: int,
    top_k: int,
    sim_thresh: float,
    bbox_thresh: float,
    time_thresh: float,
    ref_indices: List[int],
    records: List[CropRecord],
    kept: List[CropRecord],
    rejected: List[CropRecord],
    tune_mode: bool = False,
    tune_metrics: Optional[dict] = None,
) -> None:
    payload: dict = {
        "track_id": track_id,
        "score_mode": "independent",
        "tune_mode": tune_mode,
        "top_k": top_k,
        "sim_thresh": sim_thresh,
        "bbox_thresh": bbox_thresh,
        "time_thresh": time_thresh,
        "ref_crops": [records[i].crop_name for i in ref_indices],
        "kept": [r.crop_name for r in kept],
        "rejected": [
            {
                "crop_name": r.crop_name,
                "sim_to_ref": round(r.sim_to_ref, 4),
                "bbox_dist": round(r.bbox_dist, 1),
                "time_gap": round(r.time_gap, 1),
                "reasons": r.reject_reasons,
            }
            for r in rejected
        ],
    }
    if tune_metrics:
        payload["tune_metrics"] = tune_metrics
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[8] JSON -> {json_path}")


def write_combined_result_json(
    json_path: Path,
    *,
    track_id: int,
    top_k: int,
    alpha: float,
    combined_thresh: float,
    ref_indices: List[int],
    records: List[CropRecord],
    kept: List[CropRecord],
    rejected: List[CropRecord],
    tune_mode: bool = False,
    tune_metrics: Optional[dict] = None,
) -> None:
    payload: dict = {
        "track_id": track_id,
        "score_mode": "combined",
        "tune_mode": tune_mode,
        "top_k": top_k,
        "alpha": alpha,
        "combined_thresh": combined_thresh,
        "ref_crops": [records[i].crop_name for i in ref_indices],
        "kept": [r.crop_name for r in kept],
        "rejected": [
            {
                "crop_name": r.crop_name,
                "sim_to_query": round(r.sim_to_query, 4),
                "mean_sim_to_ref": round(r.sim_to_ref, 4),
                "combined_score": round(r.combined_score, 4),
                "reasons": r.reject_reasons,
            }
            for r in rejected
        ],
    }
    if tune_metrics:
        payload["tune_metrics"] = tune_metrics
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[8] JSON -> {json_path}")


def evaluate_params(
    records: List[CropRecord],
    ref_indices: List[int],
    *,
    sim_thresh: float,
    bbox_thresh: float,
    time_thresh: float,
    gt_rejected: set[str],
) -> dict:
    kept, rejected = apply_filter(
        records,
        sim_thresh=sim_thresh,
        bbox_thresh=bbox_thresh,
        time_thresh=time_thresh,
        verbose=False,
    )
    pred_rej = {r.crop_name for r in rejected}
    tp = len(gt_rejected & pred_rej)
    fp = len(pred_rej - gt_rejected)
    fn = len(gt_rejected - pred_rej)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "sim_thresh": sim_thresh,
        "bbox_thresh": bbox_thresh,
        "time_thresh": time_thresh,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kept": kept,
        "rejected": rejected,
    }


def run_tune_mode(
    records: List[CropRecord],
    ref_indices: List[int],
    *,
    track_id: int,
    query_result_json: Path,
    query_image: Path,
    output_dir: Path,
) -> None:
    gt = GT_REJECTED_TRACK24
    print(f"[tune] Ground truth rejected ({len(gt)}): {sorted(gt)}")

    sim_grid = [0.65, 0.70, 0.75, 0.80, 0.85]
    bbox_grid = [200, 300, 400, 500]
    time_grid = [30, 60, 120, 300]

    results: List[dict] = []
    total = len(sim_grid) * len(bbox_grid) * len(time_grid)
    print(f"[tune] 搜尋 {total} 組參數 …")

    # 備份 ref 標記
    ref_names = {records[i].crop_name for i in ref_indices}

    for sim_t in sim_grid:
        for bbox_t in bbox_grid:
            for time_t in time_grid:
                for rec in records:
                    rec.is_ref = rec.crop_name in ref_names
                    rec.kept = True
                    rec.reject_reasons = []
                row = evaluate_params(
                    records,
                    ref_indices,
                    sim_thresh=sim_t,
                    bbox_thresh=bbox_t,
                    time_thresh=time_t,
                    gt_rejected=gt,
                )
                results.append(row)

    results.sort(key=lambda r: (-r["f1"], -r["precision"], -r["recall"]))
    print("\n=== Top-10 F1 ===")
    print(f"{'Rank':>4}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'sim':>5}  {'bbox':>5}  {'time':>5}  {'TP':>2} {'FP':>2} {'FN':>2}")
    for rank, row in enumerate(results[:10], 1):
        print(
            f"{rank:>4}  {row['f1']:>6.3f}  {row['precision']:>6.3f}  {row['recall']:>6.3f}  "
            f"{row['sim_thresh']:>5.2f}  {row['bbox_thresh']:>5.0f}  {row['time_thresh']:>5.0f}  "
            f"{row['tp']:>2}  {row['fp']:>2}  {row['fn']:>2}"
        )

    best = results[0]
    print("\n建議參數：")
    print(
        f"  python3 BoT-SORT-K809/filter_tracklet_crops.py \\\n"
        f"    --query-result-json {query_result_json} \\\n"
        f"    --query-image {query_image} \\\n"
        f"    --track-id {track_id} \\\n"
        f"    --score-mode independent \\\n"
        f"    --sim-thresh {best['sim_thresh']} \\\n"
        f"    --bbox-thresh {int(best['bbox_thresh'])} \\\n"
        f"    --time-thresh {int(best['time_thresh'])}"
    )

    png_path, json_path = filter_result_paths(output_dir, track_id)
    save_filter_collage(
        kept=best["kept"],
        rejected=best["rejected"],
        track_id=track_id,
        sim_thresh=best["sim_thresh"],
        bbox_thresh=best["bbox_thresh"],
        time_thresh=best["time_thresh"],
        out_path=png_path,
    )
    write_independent_result_json(
        json_path,
        track_id=track_id,
        top_k=len(ref_indices),
        sim_thresh=best["sim_thresh"],
        bbox_thresh=best["bbox_thresh"],
        time_thresh=best["time_thresh"],
        ref_indices=ref_indices,
        records=records,
        kept=best["kept"],
        rejected=best["rejected"],
        tune_mode=True,
        tune_metrics={
            "f1": round(best["f1"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
        },
    )


def _reset_records_for_tune(
    records: List[CropRecord],
    ref_indices: List[int],
) -> None:
    ref_names = {records[i].crop_name for i in ref_indices}
    for rec in records:
        rec.is_ref = rec.crop_name in ref_names
        rec.kept = rec.is_ref
        rec.reject_reasons = []
        rec.combined_score = 0.0


def evaluate_params_combined(
    records: List[CropRecord],
    ref_indices: List[int],
    *,
    alpha: float,
    combined_thresh: float,
    gt_rejected: set[str],
) -> dict:
    _reset_records_for_tune(records, ref_indices)
    compute_mean_sim_to_ref(records, ref_indices)
    compute_combined_scores(records, alpha)
    kept, rejected = apply_filter_combined(
        records, combined_thresh=combined_thresh, verbose=False
    )
    pred_rej = {r.crop_name for r in rejected}
    tp = len(gt_rejected & pred_rej)
    fp = len(pred_rej - gt_rejected)
    fn = len(gt_rejected - pred_rej)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "alpha": alpha,
        "combined_thresh": combined_thresh,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kept": kept,
        "rejected": rejected,
    }


def run_tune_mode_combined(
    records: List[CropRecord],
    ref_indices: List[int],
    *,
    track_id: int,
    query_result_json: Path,
    query_image: Path,
    output_dir: Path,
) -> None:
    gt = GT_REJECTED_TRACK24
    print(f"[tune-combined] Ground truth rejected ({len(gt)}): {sorted(gt)}")

    alpha_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    thresh_grid = [0.78, 0.80, 0.82, 0.84, 0.86, 0.88]
    results: List[dict] = []
    total = len(alpha_grid) * len(thresh_grid)
    print(f"[tune-combined] 搜尋 {total} 組 (alpha × combined_thresh) …")

    for alpha in alpha_grid:
        for thresh in thresh_grid:
            row = evaluate_params_combined(
                records,
                ref_indices,
                alpha=alpha,
                combined_thresh=thresh,
                gt_rejected=gt,
            )
            results.append(row)

    results.sort(key=lambda r: (-r["f1"], -r["precision"], -r["recall"]))
    print("\n=== Top-10 F1 (combined) ===")
    print(
        f"{'Rank':>4}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  "
        f"{'alpha':>5}  {'thresh':>6}  {'TP':>2} {'FP':>2} {'FN':>2}"
    )
    for rank, row in enumerate(results[:10], 1):
        print(
            f"{rank:>4}  {row['f1']:>6.3f}  {row['precision']:>6.3f}  {row['recall']:>6.3f}  "
            f"{row['alpha']:>5.1f}  {row['combined_thresh']:>6.2f}  "
            f"{row['tp']:>2}  {row['fp']:>2}  {row['fn']:>2}"
        )

    best = results[0]
    print("\n建議參數：")
    print(
        f"  python3 BoT-SORT-K809/filter_tracklet_crops.py \\\n"
        f"    --query-result-json {query_result_json} \\\n"
        f"    --query-image {query_image} \\\n"
        f"    --track-id {track_id} \\\n"
        f"    --score-mode combined \\\n"
        f"    --alpha {best['alpha']} \\\n"
        f"    --combined-thresh {best['combined_thresh']}"
    )

    _reset_records_for_tune(records, ref_indices)
    compute_mean_sim_to_ref(records, ref_indices)
    compute_combined_scores(records, best["alpha"])
    kept, rejected = apply_filter_combined(
        records, combined_thresh=best["combined_thresh"], verbose=True
    )
    png_path, json_path = filter_result_paths(output_dir, track_id)
    save_filter_collage_combined(
        kept=kept,
        rejected=rejected,
        track_id=track_id,
        alpha=best["alpha"],
        combined_thresh=best["combined_thresh"],
        out_path=png_path,
    )
    write_combined_result_json(
        json_path,
        track_id=track_id,
        top_k=len(ref_indices),
        alpha=best["alpha"],
        combined_thresh=best["combined_thresh"],
        ref_indices=ref_indices,
        records=records,
        kept=kept,
        rejected=rejected,
        tune_mode=True,
        tune_metrics={
            "f1": round(best["f1"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tracklet intra-filter (CLIP-ReID + 時空)")
    p.add_argument("--query-result-json", type=Path, required=True)
    p.add_argument("--query-image", type=Path, required=True)
    p.add_argument("--track-id", type=int, required=True)
    p.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--embed-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--sim-thresh", type=float, default=0.75)
    p.add_argument("--bbox-thresh", type=float, default=300.0)
    p.add_argument("--time-thresh", type=float, default=60.0)
    p.add_argument(
        "--score-mode",
        choices=["independent", "combined"],
        default="independent",
        help="independent：三獨立門檻；combined：alpha*sim_q + (1-alpha)*mean_ref",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="combined 模式權重（sim_to_query）",
    )
    p.add_argument(
        "--combined-thresh",
        type=float,
        default=0.82,
        help="combined 模式剔除門檻",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tune-mode", action="store_true")
    p.add_argument(
        "--video-id",
        type=str,
        default="",
        help="mapping 篩選 video_id，預設從 query_result 讀取",
    )
    add_common_model_args(p)
    p.set_defaults(
        repo_root=str(CLIP_REID_ROOT),
        config_file=str(CLIP_REID_ROOT / "configs" / "person" / "vit_clipreid.yml"),
        weight=str(CLIP_REID_ROOT / "pretrained" / "Market1501_clipreid_ViT-B-16_60.pth"),
        num_classes=751,
        camera_num=6,
        view_num=1,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    query_json = args.query_result_json.resolve()
    query_image = resolve_path(str(args.query_image))
    if not query_json.is_file():
        raise SystemExit(f"query_result.json 不存在: {query_json}")
    if not query_image.is_file():
        raise SystemExit(f"query 圖片不存在: {query_image}")

    print("[1] 讀取 query_result …")
    payload = json.loads(query_json.read_text(encoding="utf-8"))
    track = find_track(payload, args.track_id)
    video_id = args.video_id or str(payload.get("video_id", "K8-09"))
    # mapping 內常為 K8-09；query 可能為 人員追蹤_20260528_K8-09
    map_vids = ["K8-09", video_id]
    if "K8-09" not in video_id:
        map_vids.append(video_id.split("_")[-1] if "_" in video_id else video_id)
    print(f"    track_id={args.track_id}  n_crops={track.get('n_crops')}  video_id={video_id}")

    print(f"[2] 載入 mapping: {args.mapping_json}")
    fallback_map = None
    if "20260528" in video_id and FALLBACK_MAPPING_20260528.is_file():
        fallback_map = FALLBACK_MAPPING_20260528
        if video_id not in map_vids:
            map_vids = list(map_vids) + [video_id]
    bbox_lookup = load_bbox_lookup(
        args.mapping_json.resolve(),
        map_vids,
        fallback_json=fallback_map,
    )
    print(f"    bbox 索引 {len(bbox_lookup)} 筆")

    records = build_records(track, bbox_lookup)
    cache_path = args.embed_cache.resolve()

    embedder = ClipReIDEmbedder(
        repo_root=args.repo_root,
        config_file=args.config_file,
        weight=args.weight,
        num_classes=args.num_classes,
        camera_num=args.camera_num,
        view_num=args.view_num,
    )
    cache = load_embedding_cache(cache_path)
    query_emb = ensure_embeddings(
        records, cache, cache_path, embedder, query_image
    )

    ref_indices = select_reference_set(records, query_emb, args.top_k)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tune_mode:
        if args.track_id != 24:
            print("[警告] tune-mode 的 GT 僅定義 track_id=24，仍會使用該 GT 評估")
        if args.score_mode == "combined":
            run_tune_mode_combined(
                records,
                ref_indices,
                track_id=args.track_id,
                query_result_json=query_json,
                query_image=query_image,
                output_dir=output_dir,
            )
        elif args.score_mode == "independent":
            compute_metrics(records, ref_indices)
            run_tune_mode(
                records,
                ref_indices,
                track_id=args.track_id,
                query_result_json=query_json,
                query_image=query_image,
                output_dir=output_dir,
            )
        return

    if args.score_mode == "independent":
        compute_metrics(records, ref_indices)
        kept, rejected = apply_filter(
            records,
            sim_thresh=args.sim_thresh,
            bbox_thresh=args.bbox_thresh,
            time_thresh=args.time_thresh,
        )
        out_png, out_json = filter_result_paths(output_dir, args.track_id)
        save_filter_collage(
            kept=kept,
            rejected=rejected,
            track_id=args.track_id,
            sim_thresh=args.sim_thresh,
            bbox_thresh=args.bbox_thresh,
            time_thresh=args.time_thresh,
            out_path=out_png,
        )
        write_independent_result_json(
            out_json,
            track_id=args.track_id,
            top_k=args.top_k,
            sim_thresh=args.sim_thresh,
            bbox_thresh=args.bbox_thresh,
            time_thresh=args.time_thresh,
            ref_indices=ref_indices,
            records=records,
            kept=kept,
            rejected=rejected,
        )

    elif args.score_mode == "combined":
        print(
            f"[4] combined 評分  alpha={args.alpha}  "
            f"combined_thresh={args.combined_thresh}"
        )
        compute_mean_sim_to_ref(records, ref_indices)
        compute_combined_scores(records, args.alpha)
        kept, rejected = apply_filter_combined(
            records,
            combined_thresh=args.combined_thresh,
        )
        out_png, out_json = filter_result_paths(output_dir, args.track_id)
        save_filter_collage_combined(
            kept=kept,
            rejected=rejected,
            track_id=args.track_id,
            alpha=args.alpha,
            combined_thresh=args.combined_thresh,
            out_path=out_png,
        )
        write_combined_result_json(
            out_json,
            track_id=args.track_id,
            top_k=args.top_k,
            alpha=args.alpha,
            combined_thresh=args.combined_thresh,
            ref_indices=ref_indices,
            records=records,
            kept=kept,
            rejected=rejected,
        )


if __name__ == "__main__":
    main()
