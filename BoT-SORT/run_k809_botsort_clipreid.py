#!/usr/bin/env python3
"""
K8-09：現成 bbox + crop → CLIP-ReID embedding → BoT-SORT 追蹤 → 拼圖。

不使用 BoT-SORT 內建 FastReID；embedding 由 ClipReIDEmbedder 外部計算後注入。

輸出：output/tracking_collage.png

範例：
  python3 run_k809_botsort_clipreid.py
  python3 run_k809_botsort_clipreid.py --limit-frames 50
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import types
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NumPy 2.x 相容（BoT-SORT 原始碼使用已移除的 np.float / np.int_）
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Paths（BoT-SORT 根目錄 + CLIP-ReID）
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BOTSORT_ROOT = REPO_ROOT / "BoT-SORT"
CLIP_REID_ROOT = REPO_ROOT / "CLIP-ReID"
OUTPUT_ROOT = REPO_ROOT.parent / "output"
sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(BOTSORT_ROOT))
sys.path.insert(0, str(CLIP_REID_ROOT))

DEFAULT_VIDEO_ID = "K8-09"
# BoT-SORT 以 global_frame 分組：同一 (segment, frame) 的所有 detection 必須共用同一 gf。
# 舊版 seg*151+frame 會在 s{N}_151 與 s{N+1}_0 碰撞；+1/+2 仍易踩邊界。
# 改用足夠大的 stride，確保任意 segment_idx×stride+frame 唯一（frame < stride 即可）。
GLOBAL_FRAME_STRIDE = 10_000


def compute_global_frame(segment_idx: int, frame: int) -> int:
    return segment_idx * GLOBAL_FRAME_STRIDE + frame

IMG_H, IMG_W = 1080, 1920
CROP_SEG_RE = re.compile(r"crop_s(\d+)_(\d+)_")

DEFAULT_MAPPING = OUTPUT_ROOT / "人員追蹤_crop_time_mapping.json"
DEFAULT_OUT = OUTPUT_ROOT / "botsort_standalone"


def default_crop_dir(video_id: str) -> Path:
    return OUTPUT_ROOT / video_id


def default_output_dir(video_id: str) -> Path:
    slug = video_id.lower().replace("-", "")
    return OUTPUT_ROOT / "embed_cache" / slug
DEFAULT_CACHE = DEFAULT_OUT / "clipreid_embeddings_cache.pkl"
DEFAULT_COLLAGE = DEFAULT_OUT / "tracking_collage.png"


# ---------------------------------------------------------------------------
# Stub FastReID（BoT-SORT 不載入內建 ReID）
# ---------------------------------------------------------------------------
def _install_fastreid_stub() -> None:
    fast_pkg = types.ModuleType("fast_reid")
    iface = types.ModuleType("fast_reid.fast_reid_interfece")

    class FastReIDInterface:
        """佔位；實際由 ExternalReIDEncoder 取代。"""

        def __init__(self, *args, **kwargs):
            self._feats: Optional[np.ndarray] = None

        def inference(self, img, dets):
            n = len(dets) if dets is not None and len(dets) else 0
            if self._feats is None or len(self._feats) != n:
                return np.ones((n, 512), dtype=np.float32)
            return self._feats

    iface.FastReIDInterface = FastReIDInterface
    fast_pkg.fast_reid_interfece = iface
    sys.modules["fast_reid"] = fast_pkg
    sys.modules["fast_reid.fast_reid_interfece"] = iface


_install_fastreid_stub()

from clipreid_crossmodal import ClipReIDEmbedder, add_common_model_args  # noqa: E402
from tracker.bot_sort import BoTSORT  # noqa: E402

from tracking_collage import save_tracking_collage  # noqa: E402


class ExternalReIDEncoder:
    def __init__(self, *args, **kwargs):
        self._feats: Optional[np.ndarray] = None

    def inference(self, img, dets):
        n = len(dets) if dets is not None and len(dets) else 0
        if self._feats is None or self._feats.shape[0] != n:
            return np.ones((n, 512), dtype=np.float32)
        return self._feats


# ---------------------------------------------------------------------------
# Step 1：讀取 JSON
# ---------------------------------------------------------------------------
def load_records(mapping_json: Path, crop_dir: Path, video_id: str) -> List[Dict[str, Any]]:
    print(f"[1] 讀取 JSON: {mapping_json}")
    with mapping_json.open(encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = []
    for seg in data.get("segments", []):
        if seg.get("video_id") != video_id:
            continue
        for crop in seg.get("crops", []):
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
    print(f"    video_id={video_id} 共 {len(records)} 筆 crop")
    return records


def group_frames(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r["global_frame"]].append(r)
    print(f"    分組為 {len(groups)} 個 frame")
    return dict(groups)


# ---------------------------------------------------------------------------
# Step 2：CLIP-ReID embeddings
# ---------------------------------------------------------------------------
def extract_clipreid_embeddings(
    records: List[Dict[str, Any]],
    embedder: ClipReIDEmbedder,
    cache_path: Path,
    *,
    batch_size: int = 64,
    batch_log: int = 200,
) -> Dict[str, np.ndarray]:
    print(f"[2] CLIP-ReID embedding（快取 {cache_path}）")
    cache: Dict[str, np.ndarray] = {}
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        print(f"    載入快取 {len(cache)} 筆")

    paths = [r["crop_path"] for r in records]
    missing = [p for p in paths if p not in cache]
    if missing:
        print(f"    待抽取 {len(missing)} 張 …")
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            feats = embedder.encode_images(batch, batch_size=len(batch))
            for p, f in zip(batch, feats):
                cache[p] = f.astype(np.float32)
            done = min(i + batch_size, len(missing))
            if done % batch_log == 0 or done == len(missing):
                print(f"    embedding {done}/{len(missing)}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(cache, f)
        print(f"    快取已寫入 {cache_path}")
    else:
        print("    全部命中快取，跳過抽取")

    dim = next(iter(cache.values())).shape[0]
    print(f"    特徵維度 D={dim}")
    return cache


# ---------------------------------------------------------------------------
# Step 3：逐幀 detections + embeddings
# ---------------------------------------------------------------------------
def build_detections(crops: List[Dict[str, Any]]) -> np.ndarray:
    """(N, 6): x1,y1,x2,y2,score,class_mult(=1) → score 為 col4*col5。"""
    if not crops:
        return np.empty((0, 6), dtype=np.float64)
    rows = []
    for c in crops:
        x1, y1, x2, y2 = c["box"]
        rows.append([x1, y1, x2, y2, c["score"], 1.0])
    return np.array(rows, dtype=np.float64)


def build_embeddings(crops: List[Dict[str, Any]], emb_cache: Dict[str, np.ndarray]) -> np.ndarray:
    if not crops:
        return np.empty((0, 0), dtype=np.float32)
    embs = np.stack([emb_cache[c["crop_path"]] for c in crops], axis=0).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-6)


def dummy_image() -> np.ndarray:
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def detection_scores(dets: np.ndarray) -> np.ndarray:
    if dets.size == 0:
        return np.empty(0)
    if dets.shape[1] == 5:
        return dets[:, 4]
    return dets[:, 4] * dets[:, 5]


def set_encoder_features(tracker: BoTSORT, dets: np.ndarray, embs: np.ndarray) -> None:
    scores = detection_scores(dets)
    keep = scores > tracker.track_high_thresh
    tracker.encoder._feats = embs[keep] if keep.any() else np.empty((0, embs.shape[1]), dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


def match_dets_to_tracks(dets_xyxy: np.ndarray, stracks, thr: float = 0.3) -> List[int]:
    n = dets_xyxy.shape[0]
    if n == 0:
        return []
    if not stracks:
        return [-1] * n
    tb = np.array([st.tlbr for st in stracks], dtype=np.float64)
    ious = iou_matrix(dets_xyxy, tb)
    assigned = [-1] * n
    used = set()
    pairs = [(ious[d, t], d, t) for d in range(n) for t in range(tb.shape[0])]
    pairs.sort(reverse=True)
    for iou, d, t in pairs:
        if iou < thr:
            break
        if assigned[d] >= 0 or t in used:
            continue
        assigned[d] = int(stracks[t].track_id)
        used.add(t)
    return assigned


def make_botsort_args(
    *,
    video_id: str,
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    proximity_thresh: float,
    appearance_thresh: float,
) -> Namespace:
    return Namespace(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        with_reid=True,
        fast_reid_config="",
        fast_reid_weights="",
        device="gpu",
        cmc_method="none",
        name=video_id,
        ablation=False,
        mot20=False,
    )


def init_botsort(*, video_id: str, **kwargs) -> BoTSORT:
    args = make_botsort_args(video_id=video_id, **kwargs)
    tracker = BoTSORT(args, frame_rate=30)
    tracker.encoder = ExternalReIDEncoder()
    return tracker


# ---------------------------------------------------------------------------
# Step 4：BoT-SORT 追蹤
# ---------------------------------------------------------------------------
def run_botsort_tracking(
    frame_groups: Dict[int, List[Dict[str, Any]]],
    emb_cache: Dict[str, np.ndarray],
    tracker: BoTSORT,
    *,
    video_id: str,
    reset_per_segment: bool,
    limit_frames: int = 0,
) -> List[Dict[str, Any]]:
    print("[3] BoT-SORT 逐幀追蹤 …")
    img = dummy_image()
    rows: List[Dict[str, Any]] = []
    prev_seg: Optional[int] = None
    frames = sorted(frame_groups.keys())
    if limit_frames > 0:
        frames = frames[:limit_frames]
        print(f"    （除錯）僅前 {limit_frames} 個 global_frame")

    for gi, gf in enumerate(frames, 1):
        crops = frame_groups[gf]
        seg = crops[0]["segment_idx"]
        if reset_per_segment and prev_seg is not None and seg != prev_seg:
            from tracker.basetrack import BaseTrack

            BaseTrack.clear_count()
            tracker = init_botsort(
                video_id=video_id,
                track_high_thresh=tracker.track_high_thresh,
                track_low_thresh=tracker.args.track_low_thresh,
                new_track_thresh=tracker.args.new_track_thresh,
                track_buffer=tracker.max_time_lost,
                match_thresh=tracker.args.match_thresh,
                proximity_thresh=tracker.proximity_thresh,
                appearance_thresh=tracker.appearance_thresh,
            )
        prev_seg = seg

        dets = build_detections(crops)
        embs = build_embeddings(crops, emb_cache)
        set_encoder_features(tracker, dets, embs)
        stracks = tracker.update(dets, img)
        tids = match_dets_to_tracks(dets[:, :4], stracks)

        if gi % 100 == 0 or gi == len(frames):
            n_act = len(stracks)
            print(f"    frame {gi}/{len(frames)} gf={gf} dets={len(crops)} active={n_act}")

        for c, tid in zip(crops, tids):
            x1, y1, x2, y2 = c["box"]
            rows.append(
                {
                    "global_frame": gf,
                    "frame": c["frame"],
                    "segment_idx": seg,
                    "track_id": tid,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": c["score"],
                    "crop_path": c["crop_path"],
                    "relative_seconds": c["relative_seconds"],
                    "absolute_timestamp": c["absolute_timestamp"],
                }
            )
    return rows


def collect_track_rows(rows: List[Dict[str, Any]]) -> List[Tuple[int, List[Dict[str, Any]]]]:
    by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tid = int(r["track_id"])
        if tid >= 0:
            by_tid[tid].append(r)
    out = [(tid, sorted(crops, key=lambda x: (x["frame"], -x["score"]))) for tid, crops in by_tid.items()]
    out.sort(key=lambda x: x[1][0]["absolute_timestamp"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BoT-SORT + CLIP-ReID 追蹤拼圖（K8-08 / K8-09 等）")
    p.add_argument("--video-id", type=str, default=DEFAULT_VIDEO_ID, help="例：K8-08、K8-09")
    p.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--crop-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--embed-cache", type=Path, default=None, help="預設 output/clipreid_embeddings_cache.pkl")
    p.add_argument("--collage-out", type=Path, default=None, help="預設 output/tracking_collage.png")
    p.add_argument("--track-high-thresh", type=float, default=0.35)
    p.add_argument("--track-low-thresh", type=float, default=0.1)
    p.add_argument("--new-track-thresh", type=float, default=0.45)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--proximity-thresh", type=float, default=0.5)
    p.add_argument("--appearance-thresh", type=float, default=0.25)
    p.add_argument("--reset-per-segment", action="store_true")
    p.add_argument("--limit-frames", type=int, default=0)
    add_common_model_args(p)
    p.set_defaults(
        config_file=str(CLIP_REID_ROOT / "configs" / "person" / "vit_clipreid.yml"),
        weight=str(CLIP_REID_ROOT / "pretrained" / "Market1501_clipreid_ViT-B-16_60.pth"),
        num_classes=751,
        camera_num=6,
        view_num=1,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_id = args.video_id
    if args.crop_dir is None:
        args.crop_dir = default_crop_dir(video_id)
    if args.output_dir is None:
        args.output_dir = default_output_dir(video_id)
    os.chdir(HERE)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.embed_cache or out_dir / "clipreid_embeddings_cache.pkl").resolve()
    collage_path = (args.collage_out or out_dir / "tracking_collage.png").resolve()

    print("[0] 初始化 ClipReIDEmbedder …")
    embedder = ClipReIDEmbedder(
        repo_root=args.repo_root,
        config_file=args.config_file,
        weight=args.weight,
        num_classes=args.num_classes,
        camera_num=args.camera_num,
        view_num=args.view_num,
    )
    print("    CLIP-ReID 就緒")

    records = load_records(args.mapping_json, args.crop_dir, video_id=video_id)
    if not records:
        raise SystemExit(f"無 {video_id} crop 記錄")

    emb_cache = extract_clipreid_embeddings(records, embedder, cache_path)
    frame_groups = group_frames(records)

    tracker = init_botsort(
        video_id=video_id,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
    )
    print(
        f"    BoT-SORT 參數: high={args.track_high_thresh} low={args.track_low_thresh} "
        f"new={args.new_track_thresh} buffer={args.track_buffer}"
    )

    rows = run_botsort_tracking(
        frame_groups,
        emb_cache,
        tracker,
        video_id=video_id,
        reset_per_segment=args.reset_per_segment,
        limit_frames=args.limit_frames,
    )

    matched = sum(1 for r in rows if r["track_id"] >= 0)
    n_tracks = len({r["track_id"] for r in rows if r["track_id"] >= 0})
    print(f"[4] 追蹤完成: 有效配對 {matched}/{len(rows)}，唯一 track {n_tracks}")

    track_rows = collect_track_rows(rows)
    print(f"[5] 產生拼圖（{len(track_rows)} 條 track）…")
    save_tracking_collage(track_rows=track_rows, out_path=collage_path)
    print(f"[完成] {collage_path}")


if __name__ == "__main__":
    main()
