# -*- coding: utf-8 -*-
"""
分段假設參與排名：0507 / 0528 重跑（B 設定）
============================================
計分不動；單路徑＋多段進同一 Softmax；內部共存矛盾作廢。
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402
from evaluate_paths import load_gt  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

OUT = OUTPUT_ROOT / "path_enum_llr"
CALIB = OUT / "calibration_gt0507.pkl"
BRIDGES = ["K8-07_1", "K8-07_93", "K8-09_96", "K8-07_139", "K8-09_167"]
CORRIDOR = ["K8-12_14", "K8-30_5"]


def precision_recall(path_tids, gt_set, n_gt):
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    return {
        "precision": (hit / n) if n else 0.0,
        "recall": (hit / float(n_gt)) if n_gt else 0.0,
        "n_hit": hit,
        "n_path": n,
        "hit_tids": [t for t in path_tids if t in gt_set],
        "missed_gt": sorted(gt_set - set(path_tids)),
    }


def run_dataset(tag: str, merge: Path, gt_path: Path) -> dict:
    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge))
    calib = pickle.loads(CALIB.read_bytes())
    gt = load_gt(gt_path)
    gt_tids = list(gt["person_tids"])
    gt_set = set(gt_tids)
    n_gt = len(gt_tids)

    (
        tracks,
        scored,
        ranked,
        n_legal,
        nodes,
        super_report,
        gate_info,
        options,
    ) = llr.run_llr(
        merge,
        calib,
        use_emb_gate_fix=True,
        use_supernode=True,
        use_node_evidence=True,
        dt_scoring=False,
        transition_prior=False,
    )

    top = ranked[0] if ranked else None
    pr = precision_recall(top["tids"], gt_set, n_gt) if top else None
    bridges = {b: (b in (top["tids"] if top else [])) for b in BRIDGES}
    corridor = {c: (c in (top["tids"] if top else [])) for c in CORRIDOR}

    top3 = []
    for i, h in enumerate(ranked[:3], 1):
        segs = []
        for seg in h.get("segments") or []:
            segs.append(
                {
                    "segment": seg["segment"],
                    "path": seg["path"],
                    "score": seg["score"],
                    "gap_after_prev_sec": seg.get("gap_after_prev_sec"),
                    "t_start": seg["t_start"],
                    "t_end": seg["t_end"],
                }
            )
        top3.append(
            {
                "rank": i,
                "score": h["score"],
                "P": h.get("path_probability"),
                "n_segments": h.get("n_segments"),
                "type": h.get("hypothesis_type"),
                "path": h.get("path"),
                "segments": segs,
            }
        )

    # 寫正式輸出
    alt = llr.best_disjoint_alternative(ranked)
    out_txt = OUT / f"{tag}_llr_out.txt"
    out_json = OUT / f"{tag}_llr_top1.json"
    out_png = OUT / f"{tag}_llr_top1_collage.png"
    llr.write_txt_report(
        out_txt,
        merge,
        tracks,
        scored,
        ranked,
        n_legal,
        alt,
        super_report=super_report,
        gate_info=gate_info,
        segments=options.get("segments"),
    )
    collage = None
    if top:
        collage = llr.render_collage_if_available(merge, top, out_png)
    summary = llr.build_summary_json(
        merge,
        scored,
        ranked,
        collage,
        alt,
        n_legal,
        len(tracks),
        super_report=super_report,
        gate_info=gate_info,
        options=options,
    )
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "dataset": tag,
        "n_gt": n_gt,
        "n_tracks": len(tracks),
        "n_legal_edges": n_legal,
        "n_scored": len(scored),
        "n_ranked": len(ranked),
        "ranking_meta": options.get("ranking_meta"),
        "single_maximal_top1": options.get("single_maximal_top1"),
        "top1": {
            "score": top["score"] if top else None,
            "P": top.get("path_probability") if top else None,
            "n_segments": top.get("n_segments") if top else None,
            "type": top.get("hypothesis_type") if top else None,
            "path": top.get("path") if top else None,
            "prec": pr["precision"] if pr else None,
            "rec": pr["recall"] if pr else None,
            "hit": pr["n_hit"] if pr else None,
            "missed_gt": pr["missed_gt"] if pr else None,
            "bridges_in_top1": bridges,
            "n_bridges": sum(1 for v in bridges.values() if v),
            "corridor_12_14_30_5": corridor,
            "tids": top["tids"] if top else None,
        },
        "top3": top3,
        "outputs": {
            "txt": str(out_txt),
            "json": str(out_json),
            "collage": str(collage) if collage else None,
        },
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    print("======== 0507 ========")
    r07 = run_dataset(
        "人員追蹤_20260507",
        QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
        OUT / "ground_truth_20260507.json",
    )
    results["0507"] = r07
    print(
        f"0507 Top-1: n_seg={r07['top1']['n_segments']}  "
        f"prec={r07['top1']['prec']:.3f} rec={r07['top1']['rec']:.3f}  "
        f"score={r07['top1']['score']:.4f}"
    )

    print("======== 0528 ========")
    r28 = run_dataset(
        "人員追蹤_20260528",
        QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
        OUT / "ground_truth_20260528.json",
    )
    results["0528"] = r28
    print(
        f"0528 Top-1: n_seg={r28['top1']['n_segments']}  "
        f"prec={r28['top1']['prec']:.3f} rec={r28['top1']['rec']:.3f}  "
        f"score={r28['top1']['score']:.4f}  "
        f"bridges={r28['top1']['n_bridges']}/5  "
        f"corridor={r28['top1']['corridor_12_14_30_5']}"
    )

    results["elapsed_sec"] = time.time() - t0
    results["protocol"] = llr.SEGMENT_RANK_NOTE
    results["settings"] = "B: dt off, prior off, EMB 0.80, supernode, node_evidence"

    out_json = OUT / "segmented_ranking_rerun.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown
    lines = [
        "# 分段假設參與排名重跑",
        "",
        f"> {llr.SEGMENT_RANK_NOTE}",
        "> B：dt off / prior off / EMB 0.80 / supernode / node_evidence",
        "",
        "## 0507（預期：單段 Top-1 回歸）",
        f"- n_segments=**{r07['top1']['n_segments']}**  type={r07['top1']['type']}",
        f"- prec={r07['top1']['prec']:.3f}  rec={r07['top1']['rec']:.3f}  "
        f"P={r07['top1']['P']:.4f}  score={r07['top1']['score']:.4f}",
        f"- path: `{r07['top1']['path']}`",
        f"- 單路徑極大正序 Top-1（對照）: `{r07['single_maximal_top1']}`",
        "",
        "### Top-3",
    ]
    for h in r07["top3"]:
        lines.append(
            f"- #{h['rank']} score={h['score']:.4f} P={h['P']:.4f} "
            f"段數={h['n_segments']} ({h['type']})"
        )
        for seg in h["segments"]:
            gap = seg.get("gap_after_prev_sec")
            gap_s = f" gap={gap:.1f}s" if gap is not None else ""
            lines.append(f"  - seg{seg['segment']}{gap_s}: `{seg['path']}`")

    lines += [
        "",
        "## 0528（預期：分段 B 登頂、五橋出局、12_14/30_5 回歸）",
        f"- n_segments=**{r28['top1']['n_segments']}**  type={r28['top1']['type']}",
        f"- prec={r28['top1']['prec']:.3f}  rec={r28['top1']['rec']:.3f}  "
        f"P={r28['top1']['P']:.4f}  score={r28['top1']['score']:.4f}",
        f"- 五橋: {r28['top1']['bridges_in_top1']} → **{r28['top1']['n_bridges']}/5**",
        f"- 走廊: {r28['top1']['corridor_12_14_30_5']}",
        f"- missed GT: {r28['top1']['missed_gt']}",
        f"- path: `{r28['top1']['path']}`",
        "",
        "### Top-3",
    ]
    for h in r28["top3"]:
        lines.append(
            f"- #{h['rank']} score={h['score']:.4f} P={h['P']:.4f} "
            f"段數={h['n_segments']} ({h['type']})"
        )
        for seg in h["segments"]:
            gap = seg.get("gap_after_prev_sec")
            gap_s = f" gap={gap:.1f}s" if gap is not None else ""
            lines.append(f"  - seg{seg['segment']}{gap_s}: `{seg['path']}`")

    lines += [
        "",
        f"排名 meta 0528: `{json.dumps(r28.get('ranking_meta'), ensure_ascii=False)}`",
        f"總耗時 {results['elapsed_sec']:.1f}s",
    ]
    md = OUT / "segmented_ranking_rerun.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"寫入 {out_json}")
    print(f"寫入 {md}")


if __name__ == "__main__":
    main()
