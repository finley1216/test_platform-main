# -*- coding: utf-8 -*-
"""
MIN_TRANSIT hop1→0 + 分段軌跡：0507 回歸 + 0528 重跑（凍結 B）
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import llr_gate_config as gates  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import path_enum_scoring as pes  # noqa: E402
from evaluate_paths import diagnose_gt_feasibility  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

OUT = OUTPUT_ROOT / "path_enum_llr"
CALIB = OUT / "calibration_gt0507.pkl"


def log(msg: str) -> None:
    print(msg, flush=True)


def precision_recall(path_tids, gt_set, n_gt: int) -> dict:
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    return {
        "n_path": n,
        "n_hit": hit,
        "precision": (hit / n) if n else 0.0,
        "recall": hit / float(n_gt),
        "hit_tids": [t for t in path_tids if t in gt_set],
        "missed_gt": sorted(gt_set - set(path_tids)),
    }


def topk(maximal, gt_set, n_gt, k=3):
    rows = []
    for i, p in enumerate(maximal[:k], 1):
        pr = precision_recall(p["tids"], gt_set, n_gt)
        rows.append(
            {
                "rank": i,
                "precision": pr["precision"],
                "recall": pr["recall"],
                "path_probability": p.get("path_probability"),
                "score": p["score"],
                "path": " -> ".join(p.get("super_labels") or p["tids"]),
                "tids": p["tids"],
                "super_labels": p.get("super_labels"),
                **pr,
            }
        )
    return rows


def run_dataset(tag: str, n_gt: int, gt_name: str) -> dict:
    merge = QUERY_FILTER_OUTPUT_ROOT / tag
    gt = json.loads((OUT / gt_name).read_text(encoding="utf-8"))
    gt_set = set(gt["person_tids"])
    assert len(gt_set) == n_gt

    with CALIB.open("rb") as f:
        calib = pickle.load(f)

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge))
    assert pes.DEFAULT_MIN_TRANSIT_HOP1 == 0.0, pes.DEFAULT_MIN_TRANSIT_HOP1
    assert pes.DEFAULT_MIN_TRANSIT_HOP2 == 6.0

    t0 = time.time()
    tracks, scored, maximal, n_legal, nodes, srep, gate, options = llr.run_llr(
        merge,
        calib,
        use_emb_gate_fix=True,
        use_supernode=True,
        use_node_evidence=True,
        dt_scoring=False,
        transition_prior=False,
    )
    elapsed = time.time() - t0
    segments = options.get("segments") or []

    alt = llr.best_disjoint_alternative(maximal)
    out_txt = OUT / f"{tag}_llr_out.txt"
    out_json = OUT / f"{tag}_llr_top1.json"
    out_super = OUT / f"{tag}_supernodes.json"
    llr.write_txt_report(
        out_txt,
        merge,
        tracks,
        scored,
        maximal,
        n_legal,
        alt,
        super_report=srep,
        gate_info=gate,
        segments=segments,
    )
    collage = None
    try:
        if maximal:
            collage = llr.render_collage_if_available(
                merge, maximal[0], OUT / f"{tag}_llr_top1_collage.png"
            )
    except Exception as e:
        log(f"collage skip: {e}")
    summary = llr.build_summary_json(
        merge,
        scored,
        maximal,
        collage,
        alt,
        n_legal,
        len(tracks),
        super_report=srep,
        gate_info=gate,
        options=options,
    )
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_super.write_text(json.dumps(srep, ensure_ascii=False, indent=2), encoding="utf-8")

    # 分段合計 GT 覆蓋
    covered = set()
    seg_eval = []
    for seg in segments:
        pr = precision_recall(seg["tids"], gt_set, n_gt)
        covered |= set(pr["hit_tids"])
        seg_eval.append(
            {
                "segment": seg["segment"],
                "path": seg["path"],
                "score": seg["score"],
                "P": seg.get("path_probability"),
                "gap_after_prev_sec": seg.get("gap_after_prev_sec"),
                "t_start": seg["t_start"],
                "t_end": seg["t_end"],
                "precision": pr["precision"],
                "recall_segment_alone": pr["recall"],
                "hit_tids": pr["hit_tids"],
                "n_candidates_maximal": seg.get("n_candidates_maximal"),
            }
        )

    # 可行性
    gates.apply_llr_emb_gates(True)
    diag = diagnose_gt_feasibility(tracks, list(gt["person_tids"]))

    return {
        "dataset": tag,
        "elapsed_sec": elapsed,
        "min_transit_hop1": float(pes.DEFAULT_MIN_TRANSIT_HOP1),
        "min_transit_hop2": float(pes.DEFAULT_MIN_TRANSIT_HOP2),
        "n_legal_edges": n_legal,
        "n_maximal": len(maximal),
        "multi_only": srep.get("multi_only"),
        "enumeration": srep.get("enumeration"),
        "top3": topk(maximal, gt_set, n_gt, 3),
        "segments": seg_eval,
        "segments_union_gt_coverage": {
            "n_hit": len(covered),
            "n_gt": n_gt,
            "recall": len(covered) / float(n_gt),
            "hit_tids": sorted(covered),
            "missed_gt": sorted(gt_set - covered),
        },
        "feasibility_max_coverable": diag["max_gt_coverable"],
        "feasibility_longest": diag["longest_feasible_path"],
        "bottlenecks": diag["bottleneck_consecutive_edges"],
        "nodes": nodes,
        "maximal": maximal,
        "tracks": tracks,
        "calib": calib,
        "options": {k: v for k, v in options.items() if k != "segments"},
        "raw_segments": segments,
    }


def diagnose_intruder(nodes, maximal_top, calib, tids=("K8-07_1", "K8-07_93")):
    """節點 LLR 與進出邊分解（診斷用，不修）。"""
    by_tid = {}
    for sn in nodes:
        for t in sn.members:
            by_tid[t.tid] = (t, sn)
    top_labels = maximal_top.get("super_labels") or []
    top_tids = set(maximal_top["tids"])
    edges = maximal_top.get("edges") or []
    node_ev = {ne["super"]: ne for ne in (maximal_top.get("node_evidence") or [])}

    out = {}
    for tid in tids:
        if tid not in by_tid:
            out[tid] = {"present": False}
            continue
        track, sn = by_tid[tid]
        ne = node_ev.get(sn.label) or llr.node_evidence(sn.sim, calib)
        # 進出邊：路徑中涉及此超節點的邊
        in_edges, out_edges = [], []
        for e in edges:
            if e.get("to_super") == sn.label or e.get("to") in sn.tids:
                in_edges.append(
                    {
                        "from": e.get("from_super") or e.get("from"),
                        "to": e.get("to_super") or e.get("to"),
                        "score": e.get("score"),
                        "LLR_emb": e.get("LLR_emb"),
                        "LLR_dt": e.get("LLR_dt"),
                        "LLR_dH": e.get("LLR_dH"),
                        "emb": e.get("emb"),
                        "dt": e.get("dt"),
                        "hop": e.get("hop"),
                        "via": e.get("via") or f"{e.get('from')}->{e.get('to')}",
                    }
                )
            if e.get("from_super") == sn.label or e.get("from") in sn.tids:
                out_edges.append(
                    {
                        "from": e.get("from_super") or e.get("from"),
                        "to": e.get("to_super") or e.get("to"),
                        "score": e.get("score"),
                        "LLR_emb": e.get("LLR_emb"),
                        "LLR_dt": e.get("LLR_dt"),
                        "LLR_dH": e.get("LLR_dH"),
                        "emb": e.get("emb"),
                        "dt": e.get("dt"),
                        "hop": e.get("hop"),
                        "via": e.get("via") or f"{e.get('from')}->{e.get('to')}",
                    }
                )
        out[tid] = {
            "present": True,
            "in_top1": tid in top_tids,
            "super": sn.label,
            "sim": track.sim,
            "span": [track.t_start, track.t_end],
            "node_evidence": {
                "enabled": ne.get("enabled"),
                "sim": ne.get("sim"),
                "raw": ne.get("raw"),
                "w": ne.get("w"),
                "score": ne.get("score"),
            },
            "in_edges": in_edges,
            "out_edges": out_edges,
            "note": "診斷校準移植；本輪不修",
        }
    return out


def main():
    log("=== 0507 回歸（B）===")
    r07 = run_dataset("人員追蹤_20260507", 11, "ground_truth_20260507.json")
    # 對照先前 B
    abl = json.loads((OUT / "ablation_dt_prior_0507.json").read_text(encoding="utf-8"))
    prev_b = abl["ablation"]["B"]["top3"][0]
    cur = r07["top3"][0]
    reg = {
        "prev_B": {
            "path": prev_b["path"],
            "precision": prev_b["precision"],
            "recall": prev_b["recall"],
            "P": prev_b["path_probability"],
            "score": prev_b["score"],
        },
        "new": {
            "path": cur["path"],
            "precision": cur["precision"],
            "recall": cur["recall"],
            "P": cur["path_probability"],
            "score": cur["score"],
        },
        "path_same": cur["path"] == prev_b["path"],
        "recall_delta": cur["recall"] - prev_b["recall"],
        "P_delta": (cur["path_probability"] or 0) - (prev_b["path_probability"] or 0),
        "precision_delta": cur["precision"] - prev_b["precision"],
        "not_worse_recall": cur["recall"] + 1e-12 >= prev_b["recall"],
        "not_worse_precision": cur["precision"] + 1e-12 >= prev_b["precision"] - 1e-9,
    }
    log(
        f"0507 Top-1 rec {prev_b['recall']:.3f}→{cur['recall']:.3f}  "
        f"P {prev_b['path_probability']:.4f}→{cur['path_probability']:.4f}  "
        f"path_same={reg['path_same']}"
    )
    log(f"0507 segments={len(r07['segments'])} union_cov={r07['segments_union_gt_coverage']}")

    log("\n=== 0528 重跑（B）===")
    r28 = run_dataset("人員追蹤_20260528", 16, "ground_truth_20260528.json")
    for row in r28["top3"]:
        log(
            f"  #{row['rank']} prec={row['precision']:.3f} rec={row['recall']:.3f} "
            f"P={row['path_probability']:.4f}  {row['path']}"
        )
    for seg in r28["segments"]:
        gap = seg["gap_after_prev_sec"]
        gap_s = f" gap={gap:.1f}s" if gap is not None else ""
        log(
            f"  seg{seg['segment']}: P={seg['P']:.4f} score={seg['score']:.2f}{gap_s}  "
            f"{seg['path']}"
        )
    log(f"分段合計覆蓋 {r28['segments_union_gt_coverage']}")

    diag_07 = diagnose_intruder(
        r28["nodes"], r28["maximal"][0], r28["calib"], tids=("K8-07_1", "K8-07_93")
    )
    log("07_1 / 07_93 診斷已寫入")

    # 去重不可序列化
    for r in (r07, r28):
        for k in ("nodes", "maximal", "tracks", "calib", "raw_segments"):
            r.pop(k, None)

    report = {
        "warning": (
            "凍結 calibration_gt0507.pkl + B；MIN_TRANSIT hop1→0 為原則修正；"
            "分段為輸出協定。0528 已用於本輪診斷，最終驗證需未見過的第三資料集。"
        ),
        "frozen": {
            "calibration": str(CALIB),
            "settings": "B: dt off, prior off, EMB 0.80, supernode, node_evidence",
            "min_transit_hop1": 0.0,
            "min_transit_hop2": 6.0,
            "min_transit_note": (
                "2026-07-15：hop1=0（相鄰視野邊界相接，無辯護下界）；"
                "hop2=6 維持"
            ),
        },
        "regression_0507": reg,
        "result_0507": r07,
        "result_0528": r28,
        "intruder_diag_0528": diag_07,
    }
    out = OUT / "rerun_min_transit_segments.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"\n寫入 {out}")
    return report


if __name__ == "__main__":
    main()
