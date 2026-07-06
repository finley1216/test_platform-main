#!/usr/bin/env python3
"""Standalone CLIP-ReID cross-modal indexing and retrieval utilities."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as T


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CropRecord:
    crop_id: int
    absolute_timestamp: Optional[str]
    source_video_folder: str
    segment: Optional[str]
    segment_start_timestamp: Optional[str]
    segment_end_timestamp: Optional[str]
    original_crop_path: str
    resolved_image_path: str
    label: Optional[str]
    score: Optional[float]
    relative_seconds: Optional[float]
    frame: Optional[int]
    box: Optional[str]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_cfg_with_dataset_fix(cfg, config_file: str) -> None:
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
    if "DATASETS" in config_data and config_data["DATASETS"] is None:
        del config_data["DATASETS"]
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(config_data, tmp, default_flow_style=False, sort_keys=False)
        tmp_path = tmp.name
    try:
        cfg.merge_from_file(tmp_path)
    finally:
        os.remove(tmp_path)


def build_transform(cfg):
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )


def load_image(path: str, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


class ClipReIDEmbedder:
    def __init__(
        self,
        repo_root: str,
        config_file: str,
        weight: str,
        num_classes: int,
        camera_num: int,
        view_num: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CLIP-ReID requires CUDA. Please run on a GPU environment.")
        repo_root = str(Path(repo_root).resolve())
        if not Path(repo_root).is_dir():
            raise FileNotFoundError(f"repo_root not found: {repo_root}")
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"config_file not found: {config_file}")
        if not Path(weight).is_file():
            raise FileNotFoundError(f"weight not found: {weight}")

        sys.path.insert(0, repo_root)
        from config import cfg  # type: ignore
        from model.make_model_clipreid import make_model  # type: ignore

        load_cfg_with_dataset_fix(cfg, config_file)
        cfg.MODEL.DEVICE_ID = str(cfg.MODEL.DEVICE_ID)
        cfg.freeze()

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
        self.device = torch.device("cuda")
        self.model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
        )
        self.model.load_param(weight)
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transform(cfg)

    def encode_images(self, img_paths: Sequence[str], batch_size: int = 64) -> np.ndarray:
        feats: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(img_paths), batch_size):
                batch_paths = img_paths[i : i + batch_size]
                batch_tensor = torch.stack(
                    [load_image(p, self.transform) for p in batch_paths], dim=0
                ).to(self.device)
                feat = self.model(batch_tensor)
                feat = torch.nn.functional.normalize(feat, dim=1)
                feats.append(feat.cpu().numpy().astype(np.float32))
        if not feats:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(feats, axis=0)

    def encode_one(self, img_path: str) -> np.ndarray:
        arr = self.encode_images([img_path], batch_size=1)
        if arr.shape[0] != 1:
            raise RuntimeError("Failed to encode query image.")
        return arr[0]


def _resolve_crop_image(data_root: Path, video_id: str, crop_path: str) -> Optional[Path]:
    rel = Path(crop_path)
    candidates = [
        data_root / video_id / rel,
        data_root / video_id / rel.name,
        data_root / video_id / "yolo_output" / "object_crops" / rel.name,
    ]
    for c in candidates:
        if c.is_file() and c.suffix.lower() in IMG_EXTS:
            return c.resolve()
    return None


def load_crop_records(data_root: Path, mapping_json: Path, folder_prefix: str = "車輛追蹤_K8-") -> List[CropRecord]:
    if not mapping_json.is_file():
        raise FileNotFoundError(f"mapping_json not found: {mapping_json}")
    payload = json.loads(mapping_json.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Invalid mapping json format: `segments` should be a list.")

    records: List[CropRecord] = []
    missing_count = 0
    for seg in segments:
        video_id = str(seg.get("video_id") or "")
        if not video_id.startswith(folder_prefix):
            continue
        crops = seg.get("crops") or []
        if not isinstance(crops, list):
            continue
        for c in crops:
            crop_id = c.get("crop_id")
            crop_path = str(c.get("crop_path") or "")
            if crop_id is None or not crop_path:
                continue
            resolved = _resolve_crop_image(data_root, video_id, crop_path)
            if resolved is None:
                missing_count += 1
                continue
            records.append(
                CropRecord(
                    crop_id=int(crop_id),
                    absolute_timestamp=c.get("absolute_timestamp"),
                    source_video_folder=video_id,
                    segment=seg.get("segment"),
                    segment_start_timestamp=seg.get("segment_start_timestamp"),
                    segment_end_timestamp=seg.get("segment_end_timestamp"),
                    original_crop_path=crop_path,
                    resolved_image_path=str(resolved),
                    label=c.get("label"),
                    score=float(c["score"]) if c.get("score") is not None else None,
                    relative_seconds=float(c["relative_seconds"]) if c.get("relative_seconds") is not None else None,
                    frame=int(c["frame"]) if c.get("frame") is not None else None,
                    box=c.get("box"),
                )
            )
    if not records:
        raise RuntimeError("No valid crop records found from mapping json.")
    if missing_count > 0:
        print(f"[Warning] {missing_count} crop images not found in data root and were skipped.")
    return records


def build_embedding_db(
    *,
    data_root: str,
    mapping_json: str,
    output_prefix: str,
    repo_root: str,
    config_file: str,
    weight: str,
    num_classes: int = 576,
    camera_num: int = 20,
    view_num: int = 8,
    batch_size: int = 64,
    folder_prefix: str = "車輛追蹤_K8-",
) -> Tuple[str, str]:
    data_root_p = Path(data_root).resolve()
    mapping_json_p = Path(mapping_json).resolve()
    out_prefix_p = Path(output_prefix).resolve()

    records = load_crop_records(data_root_p, mapping_json_p, folder_prefix=folder_prefix)
    image_paths = [r.resolved_image_path for r in records]
    print(f"[Info] valid crops for indexing: {len(image_paths)}")

    embedder = ClipReIDEmbedder(
        repo_root=repo_root,
        config_file=config_file,
        weight=weight,
        num_classes=num_classes,
        camera_num=camera_num,
        view_num=view_num,
    )
    embeddings = embedder.encode_images(image_paths, batch_size=batch_size)
    if embeddings.shape[0] != len(records):
        raise RuntimeError(
            f"Embedding size mismatch: vectors={embeddings.shape[0]}, records={len(records)}"
        )

    emb_path = out_prefix_p.with_suffix(".npy")
    meta_path = out_prefix_p.with_name(out_prefix_p.name + "_meta.json")
    ensure_parent(emb_path)
    np.save(emb_path, embeddings.astype(np.float32))
    meta_payload = {
        "count": len(records),
        "dim": int(embeddings.shape[1]),
        "embedding_file": str(emb_path),
        "records": [asdict(r) for r in records],
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(emb_path), str(meta_path)


def load_embedding_db(embedding_npy: str, metadata_json: str) -> Tuple[np.ndarray, List[Dict]]:
    emb = np.load(embedding_npy).astype(np.float32)
    payload = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if emb.ndim != 2:
        raise ValueError("Embedding npy must be 2D matrix.")
    if len(records) != emb.shape[0]:
        raise ValueError(
            f"Record count mismatch: records={len(records)} vectors={emb.shape[0]}"
        )
    return emb, records


def find_similar_objects(
    query_image_path: str,
    top_k: int,
    embedding_npy: str,
    metadata_json: str,
    repo_root: str,
    config_file: str,
    weight: str,
    num_classes: int = 576,
    camera_num: int = 20,
    view_num: int = 8,
) -> List[Dict]:
    query_path = Path(query_image_path).resolve()
    if not query_path.is_file():
        raise FileNotFoundError(f"query image not found: {query_path}")

    emb, records = load_embedding_db(embedding_npy, metadata_json)
    embedder = ClipReIDEmbedder(
        repo_root=repo_root,
        config_file=config_file,
        weight=weight,
        num_classes=num_classes,
        camera_num=camera_num,
        view_num=view_num,
    )
    q = embedder.encode_one(str(query_path)).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    g = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    scores = g @ q

    k = min(max(1, int(top_k)), len(records))
    idx = np.argsort(-scores)[:k]
    out: List[Dict] = []
    for rank, i in enumerate(idx, start=1):
        rec = dict(records[int(i)])
        rec["rank"] = rank
        rec["similarity"] = float(scores[int(i)])
        out.append(rec)
    return out


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo_root",
        default="/mnt/10THDD/M133040024/SSD/ASE/CLIP-ReID",
        help="Path to CLIP-ReID project root.",
    )
    parser.add_argument(
        "--config_file",
        default="/mnt/10THDD/M133040024/SSD/ASE/CLIP-ReID/configs/veri/vit_prom.yml",
        help="Path to CLIP-ReID config yml.",
    )
    parser.add_argument(
        "--weight",
        default="/mnt/10THDD/M133040024/SSD/ASE/CLIP-ReID/pretrained/ViT_CLIP_ReID_SIE_OLP_VeRi.pth",
        help="Path to CLIP-ReID checkpoint.",
    )
    parser.add_argument("--num_classes", type=int, default=576)
    parser.add_argument("--camera_num", type=int, default=20)
    parser.add_argument("--view_num", type=int, default=8)
