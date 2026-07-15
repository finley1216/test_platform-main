# -*- coding: utf-8 -*-
"""
OUT-OF-SAMPLE 0528 凍結跑分（拓撲補登後）
拓撲補登＝場地資訊（2026-07-15 使用者場地配置），非 0528 調參。
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import llr_gate_config as gates  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import path_enum_scoring as pes  # noqa: E402
from evaluate_paths import diagnose_gt_feasibility, write_diagnose_txt  # noqa: E402
from run_b_exact_viz import score_labeled_path  # noqa: E402
from run_oos_0528_diagnostics import (  # noqa: E402
    diagnose_09_10,
    enrich_overlaps,
    topology_inventory,
)

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

MERGE = QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528"
OUT_ROOT = OUTPUT_ROOT / "path_enum_llr"
OLD_OUT = OUTPUT_ROOT / "path_enum"
CALIB = OUT_ROOT / "calibration_gt0507.pkl"
GT_PATH = OUT_ROOT / "ground_truth_20260528.json"
N_GT = 16


def log(msg: str) -> None:
    print(msg, flush=True)


def precision_recall(path_tids: list[str], gt_set: set[str]) -> dict:
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    return {
        "n_path": n,
        "n_hit": hit,
        "precision": (hit / n) if n else 0.0,
        "recall": hit / float(N_GT),
        "hit_tids": [t for t in path_tids if t in gt_set],
        "missed_gt": sorted(gt_set - set(path_tids)),
    }


def topk_rows(maximal, gt_set, k=3):
    rows = []
    for i, p in enumerate(maximal[:k], 1):
        pr = precision_recall(p["tids"], gt_set)
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
                "n_hit": pr["n_hit"],
                "n_path": pr["n_path"],
                "hit_tids": pr["hit_tids"],
                "missed_gt": pr["missed_gt"],
            }
        )
    return rows


def eval_from_old_json(path: Path, gt_set: set[str]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    maximal = []
    for p in data.get("top10_paths") or []:
        maximal.append(
            {
                "tids": p["tids"],
                "score": p["score"],
                "path_probability": p.get("path_probability"),
            }
        )
    # Softmax over top10 only (報告用)；若 top1.json 無全量 P，用 top10 近似
    if maximal and maximal[0].get("path_probability") is None:
        scores = [p["score"] for p in maximal]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps) or 1.0
        for p, e in zip(maximal, exps):
            p["path_probability"] = e / z
    return {
        "n_paths_all": data.get("n_paths_all"),
        "n_paths_maximal": data.get("n_paths_maximal"),
        "emb_edge_min": data.get("emb_edge_min"),
        "top3": topk_rows(maximal, gt_set, 3),
        "json": str(path),
        "note": "評估自 top1.json 的 top10；P 若缺則對 top10 Softmax",
    }


def verify_topology() -> dict:
    topo = topology_inventory()
    needed_adj = {
        tuple(sorted(("K8-08", "K8-09"))),
        tuple(sorted(("K8-09", "K8-10"))),
        tuple(sorted(("K8-10", "K8-12"))),
        tuple(sorted(("K8-12", "K8-30"))),
    }
    adj = {tuple(sorted(p)) for p in pes.ADJACENT}
    ov = pes.OVERLAP_PAIRS
    ok_adj = needed_adj.issubset(adj)
    ok_ov = ov.get(tuple(sorted(("K8-09", "K8-10")))) == 3.0
    return {
        "ok": ok_adj and ok_ov,
        "needed_adj_present": {f"{a}|{b}": (a, b) in adj for a, b in needed_adj},
        "overlap_09_10": ov.get(tuple(sorted(("K8-09", "K8-10")))),
        "topology_gaps": topo["topology_gaps_blocking_scoring"],
        "can_run_scoring": topo["can_run_scoring"] and ok_adj and ok_ov,
        "inventory": topo,
    }


def run_old(gt_set: set[str], *, reuse: bool) -> dict:
    out_json = OLD_OUT / f"{MERGE.name}_top1.json"
    if reuse and out_json.is_file():
        log(f"重用舊法結果：{out_json}")
        # 仍補 Softmax P（若上次有寫入完整 maximal 的 JSON 僅 top10）
        # 重新從檔案評估；另從先前 stdout 知 top1 P≈0.88（全量 maximal Softmax）
        # 為正確 P，若有快取 results 則更好；此處可選重跑 Softmax——需完整 maximal。
        # 上次 run 已把 path_probability 寫入？build_summary 的 top10 不含 P。
        # 用先前腳本印出的全量 Softmax 值：改為快速重跑只取 maximal Softmax（枚舉已證 ~數分鐘）
        pass

    pes.SIM_MIN = 0.85
    pes.EMB_EDGE_MIN = float(gates.ORIGINAL_EMB_EDGE_MIN)
    pes.EMB_HIST_MIN = float(gates.ORIGINAL_EMB_HIST_MIN)
    pes.configure_for_input(str(MERGE))
    OLD_OUT.mkdir(parents=True, exist_ok=True)

    log("舊法 enum…")
    t0 = time.time()
    scored, maximal = pes.run(str(MERGE), ground_truth_tids=None)
    log(f"舊法 enum 完成 {time.time()-t0:.1f}s  maximal={len(maximal)}")

    if maximal:
        scores = [p["score"] for p in maximal]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps) or 1.0
        for p, e in zip(maximal, exps):
            p["path_probability"] = float(e / z)

    collage = None
    try:
        if maximal:
            collage = pes.render_top1_collage(
                MERGE, maximal[0], OLD_OUT / f"{MERGE.name}_top1_collage.png"
            )
    except Exception as e:
        log(f"舊法 collage 略過：{e}")

    summary = pes.build_summary(MERGE, scored, maximal, collage)
    # 把 Softmax P 寫進 top10
    for i, p in enumerate(maximal[:10], 1):
        if i - 1 < len(summary.get("top10_paths") or []):
            summary["top10_paths"][i - 1]["path_probability"] = p.get("path_probability")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "n_paths_all": summary.get("n_paths_all"),
        "n_paths_maximal": len(maximal),
        "emb_edge_min": float(pes.EMB_EDGE_MIN),
        "top3": topk_rows(maximal, gt_set, 3),
        "json": str(out_json),
        "collage": str(collage) if collage else None,
        "elapsed_sec": time.time() - t0,
    }


def run_llr_b(gt_set: set[str]) -> dict:
    with CALIB.open("rb") as f:
        calib = pickle.load(f)

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(MERGE))
    log("LLR-B enum…")
    t0 = time.time()
    (
        tracks,
        scored,
        maximal,
        n_legal_edges,
        nodes,
        super_report,
        gate_info,
        options,
    ) = llr.run_llr(
        MERGE,
        calib,
        use_emb_gate_fix=True,
        use_supernode=True,
        use_node_evidence=True,
        dt_scoring=False,
        transition_prior=False,
    )
    log(f"LLR-B 完成 {time.time()-t0:.1f}s  legal={n_legal_edges} maximal={len(maximal)}")

    tag = MERGE.name
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / f"{tag}_llr_top1.json"
    out_super = OUT_ROOT / f"{tag}_supernodes.json"
    out_png = OUT_ROOT / f"{tag}_llr_top1_collage.png"

    alt = llr.best_disjoint_alternative(maximal)
    llr.write_txt_report(
        OUT_ROOT / f"{tag}_llr_out.txt",
        MERGE,
        tracks,
        scored,
        maximal,
        n_legal_edges,
        alt,
        super_report=super_report,
        gate_info=gate_info,
    )
    collage = None
    try:
        if maximal:
            collage = llr.render_collage_if_available(MERGE, maximal[0], out_png)
    except Exception as e:
        log(f"LLR collage 略過：{e}")

    summary = llr.build_summary_json(
        MERGE,
        scored,
        maximal,
        collage,
        alt,
        n_legal_edges,
        len(tracks),
        super_report=super_report,
        gate_info=gate_info,
        options=options,
    )
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_super.write_text(json.dumps(super_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "n_tracks": len(tracks),
        "n_supernodes": len(nodes),
        "n_legal_edges": n_legal_edges,
        "n_paths_all": summary.get("n_paths_all"),
        "n_paths_maximal": len(maximal),
        "enumeration": (super_report.get("enumeration") or {}),
        "options": options,
        "gate_info": gate_info,
        "top3": topk_rows(maximal, gt_set, 3),
        "json": str(out_json),
        "collage": str(collage) if collage else None,
        "maximal": maximal,
        "tracks": tracks,
        "calib": calib,
        "nodes": nodes,
        "super_report": super_report,
        "elapsed_sec": time.time() - t0,
    }


def find_best_gt_path(maximal, gt_set):
    best = None
    for i, p in enumerate(maximal, 1):
        pr = precision_recall(p["tids"], gt_set)
        cand = {
            "rank": i,
            "score": p["score"],
            "path_probability": p.get("path_probability"),
            "path": " -> ".join(p.get("super_labels") or p["tids"]),
            "tids": p["tids"],
            "super_labels": p.get("super_labels"),
            **pr,
        }
        if best is None:
            best = cand
            continue
        key = (pr["precision"], pr["recall"], -pr["n_path"])
        bkey = (best["precision"], best["recall"], -best["n_path"])
        if key > bkey:
            best = cand
    return best


def main():
    t0 = time.time()
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_tids = list(gt["person_tids"])
    assert len(gt_tids) == N_GT
    gt_set = set(gt_tids)

    log("=== 拓撲驗證 ===")
    topo_check = verify_topology()
    log(json.dumps({k: v for k, v in topo_check.items() if k != "inventory"}, ensure_ascii=False, indent=2))
    if not topo_check["can_run_scoring"]:
        raise SystemExit("拓撲仍有缺口")

    skip_old = "--reuse-old" in sys.argv
    log("\n=== 舊法跑分 ===")
    if skip_old and (OLD_OUT / f"{MERGE.name}_top1.json").is_file():
        # 仍需全量 Softmax P：重跑舊法（不可省，因 top1.json 只有 top10）
        log("注意：為取得全量 Softmax P，仍重跑舊法 enum")
    old_res = run_old(gt_set, reuse=False)
    for r in old_res["top3"]:
        P = r["path_probability"]
        log(
            f"  #{r['rank']} prec={r['precision']:.3f} rec={r['recall']:.3f} "
            f"P={P:.4e}  {r['path']}"
        )

    log("\n=== LLR-B 跑分 ===")
    llr_res = run_llr_b(gt_set)
    for r in llr_res["top3"]:
        P = r["path_probability"]
        Ps = f"{P:.4f}" if P is not None else "—"
        log(
            f"  #{r['rank']} prec={r['precision']:.3f} rec={r['recall']:.3f} "
            f"P={Ps}  {r['path']}"
        )

    best_beam = find_best_gt_path(llr_res["maximal"], gt_set)
    log(
        f"beam 最佳 GT 對齊：#{best_beam['rank']} prec={best_beam['precision']:.3f} "
        f"rec={best_beam['recall']:.3f}  {best_beam['path']}"
    )

    log("\n=== 可行性重診斷 ===")
    gates.apply_llr_emb_gates(True)
    pes.configure_for_input(str(MERGE))
    tracks = pes.load_tracks(str(MERGE))
    diag = diagnose_gt_feasibility(tracks, gt_tids)
    enrich_overlaps(diag)
    write_diagnose_txt(diag, OUT_ROOT / "gt_feasibility_20260528.txt")
    diag_09_10 = diagnose_09_10(tracks)

    long_tids = diag["longest_feasible_path"]["tids"]
    tid_to_label = {}
    for sn in llr_res["nodes"]:
        for tid in sn.tids:
            tid_to_label[tid] = sn.label
    labels = []
    for tid in long_tids:
        lab = tid_to_label.get(tid, tid)
        if not labels or labels[-1] != lab:
            labels.append(lab)

    log(f"最長可行：{diag['max_gt_coverable']}/{diag['n_gt']}  labels={' -> '.join(labels)}")
    exact = score_labeled_path(
        llr_res["nodes"], labels, llr_res["calib"], dt_scoring=False, transition_prior=False
    )
    # 在 beam maximal 找同標籤排名
    rank_beam = None
    for i, p in enumerate(llr_res["maximal"], 1):
        if p.get("super_labels") == labels:
            rank_beam = i
            break
    exact_info = {
        "labels": labels,
        "score_labeled": {
            k: exact.get(k)
            for k in ("ok", "score", "reason", "super_labels", "tids")
        },
        "rank_in_beam_maximal": rank_beam,
        "note": "不跑全量 leaf DFS（合法邊>80；beam 同 0507）。僅指定鏈精確計分＋beam 內排名。",
    }
    log(
        f"指定鏈精確計分 ok={exact.get('ok')} score={exact.get('score')} "
        f"beam_rank=#{rank_beam}"
    )

    abl = json.loads((OUT_ROOT / "ablation_dt_prior_0507.json").read_text(encoding="utf-8"))
    b0507 = abl["ablation"]["B"]["top3"][0]
    ev = json.loads((OUT_ROOT / "evaluate_20260507.json").read_text(encoding="utf-8"))
    old0507 = ev["old"]["top10"][0]

    report = {
        "warning": "OUT-OF-SAMPLE 0528；calibration_gt0507.pkl 未經 0528 調整；拓撲補登為場地資訊更新。",
        "frozen": {
            "calibration": str(CALIB),
            "settings": "B: dt-scoring off, transition-prior off, EMB 0.80, supernode, node_evidence",
            "no_0528_tuning": True,
            "out_of_sample": True,
            "topology_update": {
                "date": "2026-07-15",
                "source": "使用者提供之場地配置",
                "PERSON_ADJACENT_added": ["K8-09↔K8-10", "K8-10↔K8-12", "K8-12↔K8-30"],
                "PERSON_OVERLAP_PAIRS_added": {"K8-09↔K8-10": "tol=3s, no H"},
                "confirmed_existing": ["K8-08↔K8-09"],
                "note": "適用所有後續資料集；非針對 0528 的調參。",
            },
        },
        "topology_check": {k: v for k, v in topo_check.items() if k != "inventory"},
        "old": old_res,
        "llr_b": {
            k: v
            for k, v in llr_res.items()
            if k not in ("maximal", "tracks", "calib", "nodes", "super_report")
        },
        "beam_best_gt_aligned": best_beam,
        "exact": exact_info,
        "feasibility": {
            "n_gt": diag["n_gt"],
            "max_gt_coverable": diag["max_gt_coverable"],
            "longest_feasible_path": diag["longest_feasible_path"],
            "uncovered_by_longest": diag["uncovered_by_longest"],
            "bottleneck_consecutive_edges": diag["bottleneck_consecutive_edges"],
            "time_overlaps_among_gt": diag["time_overlaps_among_gt"],
            "consecutive_edge_checks": diag["consecutive_edge_checks"],
            "gt_sorted_by_t_start": diag["gt_sorted_by_t_start"],
            "before_topology_update": {"max_gt_coverable": 10, "n_gt": 16},
        },
        "task3_09_10": diag_09_10,
        "compare_0507": {
            "old_top1": {
                "precision": old0507["precision"],
                "recall": old0507["recall"],
                "P": old0507.get("path_probability"),
                "path": old0507["path"],
            },
            "llr_b_top1": {
                "precision": b0507["precision"],
                "recall": b0507["recall"],
                "P": b0507["path_probability"],
                "path": b0507["path"],
            },
        },
        "elapsed_sec": time.time() - t0,
    }

    out_json = OUT_ROOT / "oos_0528_results.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_ROOT / "oos_0528_diagnostics.json").write_text(
        json.dumps(
            {
                "dataset": "人員追蹤_20260528",
                "frozen": report["frozen"],
                "topology": topo_check["inventory"],
                "feasibility": diag,
                "task3_09_10": diag_09_10,
                "scoring_skipped": False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    log(f"\n完成：{out_json}")
    log(f"覆蓋上限：{diag['max_gt_coverable']}/{diag['n_gt']}")
    log(f"09_142 SN：{diag_09_10['supernode_membership'].get('K8-09_142')}")
    log(f"10_32 SN：{diag_09_10['supernode_membership'].get('K8-10_32')}")
    return report


if __name__ == "__main__":
    main()
