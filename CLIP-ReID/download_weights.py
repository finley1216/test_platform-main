#!/usr/bin/env python3
"""按需下載 CLIP-ReID 預訓練權重（人員 / 車輛）。"""
from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PRETRAINED_DIR = HERE / "pretrained"

# 官方 README「Trained models」Google Drive 連結
# https://github.com/Syliz517/CLIP-ReID
GDRIVE_PERSON_MARKET = "1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"  # ViT-CLIP-ReID Market
GDRIVE_VEHICLE_VERI_SIE = "1vb-mMGp7q_aqAB1U_uAGsHZ1U9HViOgE"  # ViT-CLIP-ReID-SIE-OLP VeRi

LabelKind = Literal["person", "vehicle"]


@dataclass(frozen=True)
class WeightSpec:
    label: LabelKind
    filename: str
    gdrive_id: str
    config_file: str
    num_classes: int
    camera_num: int
    view_num: int


PERSON_WEIGHT = WeightSpec(
    label="person",
    filename="Market1501_clipreid_ViT-B-16_60.pth",
    gdrive_id=GDRIVE_PERSON_MARKET,
    config_file="configs/person/vit_clipreid.yml",
    num_classes=751,
    camera_num=6,
    view_num=1,
)

VEHICLE_WEIGHT = WeightSpec(
    label="vehicle",
    filename="ViT_CLIP_ReID_SIE_OLP_VeRi.pth",
    gdrive_id=GDRIVE_VEHICLE_VERI_SIE,
    config_file="configs/veri/vit_prom.yml",
    num_classes=576,
    camera_num=20,
    view_num=8,
)

WEIGHTS = {
    "person": PERSON_WEIGHT,
    "vehicle": VEHICLE_WEIGHT,
}


def _legacy_pretrained_dir() -> Optional[Path]:
    """開發機上舊 ASE 佈局：../CLIP-ReID/pretrained。"""
    legacy = REPO_ROOT.parent / "CLIP-ReID" / "pretrained"
    return legacy if legacy.is_dir() else None


def _download_gdrive(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    session_url = "https://drive.google.com/uc?export=download"
    with urlopen(f"{session_url}&id={file_id}") as resp:
        data = resp.read()
        token = None
        for line in data.decode("utf-8", errors="ignore").splitlines():
            m = re.search(r"confirm=([0-9A-Za-z_]+)", line)
            if m:
                token = m.group(1)
                break
    if token:
        with urlopen(f"{session_url}&id={file_id}&confirm={token}") as resp:
            data = resp.read()
    if len(data) < 1_000_000:
        raise RuntimeError(
            f"下載 {dest.name} 失敗（檔案過小，可能為 Google Drive 確認頁）。"
            f"請手動下載後放到 {dest}"
        )
    dest.write_bytes(data)


def ensure_weight(spec: WeightSpec, *, verbose: bool = True) -> Path:
    dest = PRETRAINED_DIR / spec.filename
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest

    legacy = _legacy_pretrained_dir()
    legacy_src = legacy / spec.filename if legacy else None
    if legacy_src and legacy_src.is_file():
        if verbose:
            print(f"[weights] 從本機舊路徑複製 {legacy_src} -> {dest}", flush=True)
        shutil.copy2(legacy_src, dest)
        return dest

    if verbose:
        print(f"[weights] 下載 {spec.filename}（Google Drive）…", flush=True)
    _download_gdrive(spec.gdrive_id, dest)
    if verbose:
        print(f"[weights] 完成 -> {dest}", flush=True)
    return dest


def ensure_for_dataset(dataset_key: str, *, verbose: bool = True) -> WeightSpec:
    if dataset_key.startswith("人員"):
        spec = PERSON_WEIGHT
    elif dataset_key.startswith("車輛"):
        spec = VEHICLE_WEIGHT
    else:
        raise ValueError(f"無法判斷資料集類型：{dataset_key}")
    ensure_weight(spec, verbose=verbose)
    return spec


def weight_path(spec: WeightSpec) -> Path:
    return PRETRAINED_DIR / spec.filename


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "person"
    if target not in WEIGHTS:
        raise SystemExit(f"用法: {Path(__file__).name} [person|vehicle]")
    ensure_weight(WEIGHTS[target])
