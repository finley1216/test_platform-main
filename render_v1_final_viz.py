# -*- coding: utf-8 -*-
"""
v1.0 最終視覺化（不改排名／計分邏輯）
====================================
0507：Top-1 單段一張
0528：Top-1 兩段同軸一張（空窗 178.3s）
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import llr_gate_config as gates  # noqa: E402
from run_b_exact_viz import score_labeled_path  # noqa: E402
from dump_topology_and_viz_0528 import render_top1_sequence  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

OUT = OUTPUT_ROOT / "path_enum_llr"
VIZ_DIR = OUT / "viz_v1.0"
CALIB = OUT / "calibration_gt0507.pkl"
FREEZE_DIR = OUT / "freeze_v1.0-segmented-ranking"


def _labels_from_seg(seg: dict) -> list[str]:
    labs = seg.get("super_labels")
    if labs:
        return list(labs)
    path = seg.get("path") or ""
    # "a -> {b,c} -> d"
    parts = []
    buf = ""
    depth = 0
    for ch in path:
        if ch == "{":
            depth += 1
            buf += ch
        elif ch == "}":
            depth -= 1
            buf += ch
        elif ch == "-" and depth == 0 and buf.endswith(" "):
            continue
        elif ch == ">" and depth == 0:
            parts.append(buf.strip().rstrip("-").strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


def rebuild_display_path(nodes, labels, calib) -> dict:
    """僅為視覺化重算邊（顯示 dt/hop）；不影響排名。"""
    r = score_labeled_path(
        nodes, labels, calib, dt_scoring=False, transition_prior=False
    )
    if not r.get("ok"):
        raise RuntimeError(f"viz rebuild failed: {r.get('reason')} labels={labels}")
    return {
        "score": r["score"],
        "tids": r["tids"],
        "super_labels": r["super_labels"],
        "edges": r["edges"],
        "node_evidence": r["node_evidence"],
        "n_segments": 1,
    }


def render_dataset(tag: str, n_gt_label: str) -> Path:
    merge = QUERY_FILTER_OUTPUT_ROOT / tag
    top_json = OUT / f"{tag}_llr_top1.json"
    gt_path = OUT / (
        "ground_truth_20260507.json"
        if "0507" in tag
        else "ground_truth_20260528.json"
    )

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge))
    gates.apply_llr_emb_gates(True)
    tracks = pes.load_tracks(str(merge))
    by_tid = {t.tid: t for t in tracks}
    nodes, _ = llr.build_supernodes(tracks)
    calib = pickle.loads(CALIB.read_bytes())
    gt_set = set(json.loads(gt_path.read_text(encoding="utf-8"))["person_tids"])

    data = json.loads(top_json.read_text(encoding="utf-8"))
    top = data["top1"]
    segs_src = top.get("segments") or data.get("segments") or []
    segs_src = sorted(segs_src, key=lambda s: int(s.get("segment") or 1))

    rebuilt_segs = []
    for seg in segs_src:
        labs = _labels_from_seg(seg)
        disp = rebuild_display_path(nodes, labs, calib)
        rebuilt_segs.append(
            {
                "segment": int(seg.get("segment") or 1),
                "path": " -> ".join(disp["super_labels"]),
                "super_labels": disp["super_labels"],
                "tids": disp["tids"],
                "score": float(seg.get("score") or disp["score"]),
                "t_start": seg.get("t_start"),
                "t_end": seg.get("t_end"),
                "gap_after_prev_sec": seg.get("gap_after_prev_sec"),
                "edges": disp["edges"],
            }
        )

    seg1 = rebuilt_segs[0]
    top1_viz = {
        "score": top.get("score"),
        "path_probability": top.get("path_probability"),
        "n_segments": top.get("n_segments") or len(rebuilt_segs),
        "super_labels": seg1["super_labels"],
        "tids": seg1["tids"],
        "edges": seg1["edges"],
    }
    extra = [s for s in rebuilt_segs if int(s["segment"]) > 1]

    n_seg = int(top1_viz["n_segments"])
    title = (
        f"{tag}  v1.0 Top-1  n_seg={n_seg}  "
        f"score={top1_viz['score']:.3f}  P={top1_viz['path_probability']:.4f}  "
        f"({n_gt_label})"
    )
    out_png = VIZ_DIR / f"{tag}_v1.0_top1_sequence.png"
    png, crop_log = render_top1_sequence(
        top1_viz,
        by_tid,
        merge,
        gt_set,
        out_png,
        segments=extra,
        title=title,
    )
    crop_txt = VIZ_DIR / f"{tag}_v1.0_top1_crop_list.txt"
    crop_txt.write_text(
        "tid\tcrop_path\n" + "\n".join(crop_log) + "\n", encoding="utf-8"
    )
    print(f"寫入 {png}")
    return png


def freeze_snapshot() -> Path:
    """資料夾快照 + 校準歸檔（不改邏輯）。"""
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    # calib
    dst_calib = FREEZE_DIR / "calibration_gt0507.pkl"
    shutil.copy2(CALIB, dst_calib)
    # reports
    for name in [
        "segmented_ranking_rerun.md",
        "segmented_ranking_rerun.json",
        "人員追蹤_20260507_llr_top1.json",
        "人員追蹤_20260528_llr_top1.json",
        "bridge_selfcalib_prior_report.md",
        "topo_ablation_0509_0709.md",
        "ground_truth_20260507.json",
        "ground_truth_20260528.json",
    ]:
        src = OUT / name
        if src.is_file():
            shutil.copy2(src, FREEZE_DIR / name)

    # key source files
    src_dir = FREEZE_DIR / "src"
    src_dir.mkdir(exist_ok=True)
    for name in [
        "path_enum_llr.py",
        "path_enum_scoring.py",
        "llr_gate_config.py",
        "calibrate_from_gt.py",
        "calibrate_self.py",
        "run_segmented_ranking_rerun.py",
        "dump_topology_and_viz_0528.py",
    ]:
        p = REPO_ROOT / name
        if p.is_file():
            shutil.copy2(p, src_dir / name)

    manifest = [
        "freeze: v1.0-segmented-ranking",
        "date: 2026-07-15",
        "settings: B (dt off, prior off) + EMB 0.80 + supernode + node_evidence",
        "min_transit_hop1: 0.0  hop2: 6.0",
        "ranking: single-path + segmented hypotheses, one softmax",
        "calib: calibration_gt0507.pkl (archived here)",
        "logic: frozen; this snapshot is archive only",
        "",
        f"calib_sha256: see files",
    ]
    (FREEZE_DIR / "VERSION.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    # checksum
    import hashlib

    h = hashlib.sha256(dst_calib.read_bytes()).hexdigest()
    (FREEZE_DIR / "calibration_gt0507.sha256").write_text(
        f"{h}  calibration_gt0507.pkl\n", encoding="utf-8"
    )
    print(f"快照：{FREEZE_DIR}")
    print(f"calib sha256={h}")
    return FREEZE_DIR


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    freeze_snapshot()
    render_dataset("人員追蹤_20260507", "0507 in-sample GT=11")
    render_dataset("人員追蹤_20260528", "0528 OOS GT=16")
    print(f"視覺化目錄：{VIZ_DIR}")


if __name__ == "__main__":
    main()
