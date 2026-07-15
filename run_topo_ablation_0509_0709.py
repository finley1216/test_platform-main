# -*- coding: utf-8 -*-
"""
拓撲敏感度消融：K8-05↔K8-09、K8-07↔K8-09
============================================
凍結：calibration_gt0507.pkl、B 設定、hop1=0。
只經 apply_person_adjacent_exclusions() 動 ADJACENT；預設不移除。
敏感度診斷，最終採用與否待場地圖確認。
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
from dump_topology_and_viz_0528 import render_top1_sequence  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

OUT = OUTPUT_ROOT / "path_enum_llr"
VIZ = OUT / "viz_topo_ablation"
CALIB = OUT / "calibration_gt0507.pkl"

PAIR_05_09 = tuple(sorted(("K8-05", "K8-09")))
PAIR_07_09 = tuple(sorted(("K8-07", "K8-09")))
FOCUS_PAIRS = {PAIR_05_09, PAIR_07_09}

ABLATIONS = {
    "T0": {"exclude": set(), "desc": "現行拓撲（基準）"},
    "T1": {"exclude": {PAIR_05_09}, "desc": "移除 05↔09"},
    "T2": {"exclude": {PAIR_07_09}, "desc": "移除 07↔09"},
    "T3": {"exclude": {PAIR_05_09, PAIR_07_09}, "desc": "兩對都移除"},
}

BRIDGES_0528 = ("K8-07_1", "K8-07_93", "K8-09_96", "K8-07_139", "K8-09_167")

DATASETS = [
    {
        "tag": "人員追蹤_20260507",
        "short": "0507",
        "gt": "ground_truth_20260507.json",
        "n_gt": 11,
    },
    {
        "tag": "人員追蹤_20260528",
        "short": "0528",
        "gt": "ground_truth_20260528.json",
        "n_gt": 16,
    },
]


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


def common_neighbors(cam_u: str, cam_v: str) -> set[str]:
    nb_u = {b if a == cam_u else a for a, b in pes.ADJACENT if cam_u in (a, b)}
    nb_v = {b if a == cam_v else a for a, b in pes.ADJACENT if cam_v in (a, b)}
    return nb_u & nb_v


def analyze_gt_pair_dependency(tracks: list, gt_tids: list[str]) -> dict:
    """
    在當前 ADJACENT 下，找出 GT→GT 合法邊對 FOCUS_PAIRS 的依賴：
      - hop1 直連正好是 05↔09 或 07↔09
      - hop2 的共同鄰居集合，使「若無某一 focus 邊」會影響可達性
    """
    by = {t.tid: t for t in tracks}
    gt_tracks = [by[t] for t in gt_tids if t in by]
    hop1_deps = []
    hop2_via = []
    all_legal = []

    for u in gt_tracks:
        for v in gt_tracks:
            if u.tid == v.tid:
                continue
            ok, reason, dt, hop, emb, h_dist = pes.edge_check(u, v)
            if not ok:
                continue
            key = tuple(sorted((u.cam, v.cam)))
            rec = {
                "from": u.tid,
                "to": v.tid,
                "cams": f"{u.cam}→{v.cam}",
                "cam_pair": list(key),
                "hop": hop,
                "dt": float(dt) if dt is not None else None,
                "emb": float(emb) if emb is not None else None,
            }
            all_legal.append(rec)
            if hop == 1 and key in FOCUS_PAIRS:
                hop1_deps.append({**rec, "depends_on": f"{key[0]}↔{key[1]}"})
            if hop == 2:
                commons = sorted(common_neighbors(u.cam, v.cam))
                # 標記 focus 邊是否參與「成為 hop2」：u/v 之一為 09，共同鄰居含 05 或 07
                via_focus = []
                for mid in commons:
                    for focus in FOCUS_PAIRS:
                        if mid in focus and (u.cam in focus or v.cam in focus):
                            # e.g. 07-09 removed → 07↔05 + 05↔09 仍可能 hop2
                            via_focus.append(
                                {
                                    "common_neighbor": mid,
                                    "focus_pair": list(focus),
                                }
                            )
                # 更直接：共同鄰居路徑是否必須用到 focus 邊
                # 07↔09 hop2 via 05: 需要 07↔05 與 05↔09 —— 若砍 05↔09 則此 mid=05 無效
                needed_edges = []
                for mid in commons:
                    e1 = tuple(sorted((u.cam, mid)))
                    e2 = tuple(sorted((v.cam, mid)))
                    for fe in FOCUS_PAIRS:
                        if e1 == fe or e2 == fe:
                            needed_edges.append(list(fe))
                if needed_edges:
                    hop2_via.append(
                        {
                            **rec,
                            "common_neighbors": commons,
                            "uses_focus_edges": needed_edges,
                        }
                    )

    return {
        "n_legal_gt_ordered_pairs": len(all_legal),
        "hop1_direct_on_focus": hop1_deps,
        "hop2_using_focus_edges": hop2_via,
        "cannot_cut_if_nonempty": bool(hop1_deps or hop2_via),
    }


def diagnose_gt_under_ablation(tracks, gt_tids, exclude_set) -> dict:
    """T0 全量 ADJ 下的依賴，以及排除後 hop/合法性變化。"""
    # baseline with full adjacent
    pes.reset_person_adjacent_exclusions()
    if pes.MODE == "person":
        pes.ADJACENT = set(pes.PERSON_ADJACENT)
    base = analyze_gt_pair_dependency(tracks, gt_tids)

    # recompute each hop1/hop2 focus-related under exclusion
    by = {t.tid: t for t in tracks}
    changes = []
    focus_edges = base["hop1_direct_on_focus"] + [
        {**e, "kind": "hop2"} for e in base["hop2_using_focus_edges"]
    ]
    # Also check ALL T0-legal GT edges for change when exclude applied
    pes.reset_person_adjacent_exclusions()
    pes.ADJACENT = set(pes.PERSON_ADJACENT)
    t0_legal = []
    gt_tracks = [by[t] for t in gt_tids if t in by]
    for u in gt_tracks:
        for v in gt_tracks:
            if u.tid == v.tid:
                continue
            ok, reason, dt, hop, emb, _ = pes.edge_check(u, v)
            if ok:
                t0_legal.append((u, v, hop, dt, emb))

    pes.apply_person_adjacent_exclusions(exclude_set)
    for u, v, hop0, dt0, emb0 in t0_legal:
        ok, reason, dt, hop, emb, _ = pes.edge_check(u, v)
        key = tuple(sorted((u.cam, v.cam)))
        related = key in FOCUS_PAIRS or hop0 == 2  # may be affected
        if (not ok) or (hop != hop0):
            # only report if focus-related or hop changed involving 05/07/09
            cams = {u.cam, v.cam}
            if related or (cams & {"K8-05", "K8-07", "K8-09"}):
                changes.append(
                    {
                        "from": u.tid,
                        "to": v.tid,
                        "cams": f"{u.cam}→{v.cam}",
                        "t0_hop": hop0,
                        "t0_dt": dt0,
                        "new_ok": bool(ok),
                        "new_hop": hop,
                        "new_reason": reason or "",
                    }
                )

    return {
        "under_full_topology": base,
        "legality_or_hop_changes_vs_T0": changes,
    }


def run_one(merge: Path, calib: dict, exclude: set, gt_set: set, n_gt: int) -> dict:
    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge))
    assert pes.DEFAULT_MIN_TRANSIT_HOP1 == 0.0
    excl_info = pes.apply_person_adjacent_exclusions(exclude)

    tracks, scored, maximal, n_legal, nodes, srep, gate, options = llr.run_llr(
        merge,
        calib,
        use_emb_gate_fix=True,
        use_supernode=True,
        use_node_evidence=True,
        dt_scoring=False,
        transition_prior=False,
    )
    top = maximal[0] if maximal else None
    pr = precision_recall(top["tids"], gt_set, n_gt) if top else None
    return {
        "exclude_info": excl_info,
        "n_legal_edges": n_legal,
        "n_maximal": len(maximal),
        "n_adjacent": len(pes.ADJACENT),
        "top1": {
            "score": top["score"] if top else None,
            "path_probability": top.get("path_probability") if top else None,
            "path": " -> ".join(top.get("super_labels") or top["tids"]) if top else None,
            "super_labels": top.get("super_labels") if top else None,
            "tids": top["tids"] if top else None,
            "edges": top.get("edges") if top else None,
            "precision": pr["precision"] if pr else None,
            "recall": pr["recall"] if pr else None,
            "missed_gt": pr["missed_gt"] if pr else None,
            "hit_tids": pr["hit_tids"] if pr else None,
        },
        "bridges_in_top1": {
            b: (b in (top["tids"] if top else [])) for b in BRIDGES_0528
        },
        "tracks": tracks,
        "nodes": nodes,
        "maximal0": top,
    }


def main():
    t_all = time.time()
    with CALIB.open("rb") as f:
        calib = pickle.load(f)

    VIZ.mkdir(parents=True, exist_ok=True)
    results = {
        "warning": (
            "拓撲敏感度消融；不改 PERSON_ADJACENT 預設。"
            "最終採用與否待使用者依場地配置圖確認。"
        ),
        "frozen": {
            "calibration": str(CALIB),
            "B": "dt off, prior off, EMB 0.80, supernode, node_evidence",
            "min_transit_hop1": 0.0,
        },
        "ablations": ABLATIONS,
        "by_dataset": {},
    }

    for ds in DATASETS:
        merge = QUERY_FILTER_OUTPUT_ROOT / ds["tag"]
        gt = json.loads((OUT / ds["gt"]).read_text(encoding="utf-8"))
        gt_set = set(gt["person_tids"])
        assert len(gt_set) == ds["n_gt"]
        log(f"\n======== {ds['tag']} ========")

        # load tracks once under full topo for dependency (meta)
        pes.SIM_MIN = 0.85
        pes.configure_for_input(str(merge))
        gates.apply_llr_emb_gates(True)
        tracks_full = pes.load_tracks(str(merge))
        by_tid = {t.tid: t for t in tracks_full}

        ds_block = {"groups": {}, "gt_dependency_T0": None}
        t0_top = None

        for tid, spec in ABLATIONS.items():
            log(f"--- {tid}: {spec['desc']} ---")
            t0 = time.time()
            run = run_one(merge, calib, spec["exclude"], gt_set, ds["n_gt"])
            dep = diagnose_gt_under_ablation(
                tracks_full, list(gt["person_tids"]), spec["exclude"]
            )
            if tid == "T0":
                ds_block["gt_dependency_T0"] = dep["under_full_topology"]
                t0_top = run["top1"]

            # 0507 regression vs T0
            reg = None
            if ds["short"] == "0507" and t0_top is not None:
                worse_rec = run["top1"]["recall"] + 1e-12 < t0_top["recall"]
                worse_prec = run["top1"]["precision"] + 1e-12 < t0_top["precision"]
                path_diff = run["top1"]["path"] != t0_top["path"]
                reg = {
                    "path_same_as_T0": not path_diff,
                    "recall_delta": run["top1"]["recall"] - t0_top["recall"],
                    "precision_delta": run["top1"]["precision"] - t0_top["precision"],
                    "P_delta": (run["top1"]["path_probability"] or 0)
                    - (t0_top["path_probability"] or 0),
                    "red_flag": bool(worse_rec or worse_prec),
                    "note": "相對本消融 T0；recall/prec 變壞=紅旗",
                }

            # viz
            if run["maximal0"]:
                png = VIZ / f"{ds['tag']}_{tid}_top1_sequence.png"
                render_top1_sequence(
                    run["maximal0"],
                    by_tid,
                    merge,
                    gt_set,
                    png,
                    segments=None,
                )
                viz_path = str(png)
            else:
                viz_path = None

            # strip heavy
            entry = {
                "desc": spec["desc"],
                "exclude": [list(p) for p in sorted(spec["exclude"])],
                "exclude_info": run["exclude_info"],
                "n_legal_edges": run["n_legal_edges"],
                "n_adjacent": run["n_adjacent"],
                "top1": {
                    k: v
                    for k, v in run["top1"].items()
                    if k != "edges"
                },
                "top1_n_edges": len(run["top1"]["edges"] or []) if run["top1"] else 0,
                "bridges_in_top1": run["bridges_in_top1"]
                if ds["short"] == "0528"
                else None,
                "gt_edge_changes_vs_full": dep["legality_or_hop_changes_vs_T0"],
                "gt_dependency_summary": {
                    "hop1_direct_on_focus_under_full": dep["under_full_topology"][
                        "hop1_direct_on_focus"
                    ]
                    if tid == "T0"
                    else ds_block["gt_dependency_T0"]["hop1_direct_on_focus"]
                    if ds_block["gt_dependency_T0"]
                    else [],
                    "n_changes": len(dep["legality_or_hop_changes_vs_T0"]),
                },
                "regression_0507": reg,
                "viz": viz_path,
                "elapsed_sec": time.time() - t0,
            }
            ds_block["groups"][tid] = entry
            top = run["top1"]
            log(
                f"  Top-1 prec={top['precision']:.3f} rec={top['recall']:.3f} "
                f"P={top['path_probability']:.4f}  {top['path']}"
            )
            if reg and reg["red_flag"]:
                log("  *** 紅旗：0507 相對 T0 變壞 ***")

        results["by_dataset"][ds["short"]] = ds_block

    # 彙總表 + 同時滿足條件
    summary_rows = []
    t0_0507 = results["by_dataset"]["0507"]["groups"]["T0"]["top1"]
    for tid in ABLATIONS:
        r07 = results["by_dataset"]["0507"]["groups"]["tid" if False else tid]["top1"]
        r28 = results["by_dataset"]["0528"]["groups"][tid]["top1"]
        bridges = results["by_dataset"]["0528"]["groups"][tid]["bridges_in_top1"]
        n_bridges = sum(1 for v in bridges.values() if v)
        reg = results["by_dataset"]["0507"]["groups"][tid]["regression_0507"]
        ok_0507 = (reg is None) or (not reg["red_flag"])
        # 橋消失：五個都不在 Top-1
        bridges_gone = n_bridges == 0
        summary_rows.append(
            {
                "group": tid,
                "desc": ABLATIONS[tid]["desc"],
                "0507_prec": r07["precision"],
                "0507_rec": r07["recall"],
                "0507_P": r07["path_probability"],
                "0528_prec": r28["precision"],
                "0528_rec": r28["recall"],
                "0528_P": r28["path_probability"],
                "0528_n_bridges_in_top1": n_bridges,
                "0528_bridges": bridges,
                "0507_not_worse": ok_0507,
                "0528_bridges_gone": bridges_gone,
                "meets_both": bool(ok_0507 and bridges_gone and tid != "T0"),
            }
        )

    results["summary_table"] = summary_rows
    results["meets_0507_ok_and_0528_bridges_gone"] = [
        r["group"] for r in summary_rows if r["meets_both"]
    ]

    # markdown report
    md_lines = []
    md_lines.append("# 拓撲敏感度消融：K8-05↔K8-09、K8-07↔K8-09\n")
    md_lines.append(
        "> 凍結 `calibration_gt0507.pkl` + B + hop1=0。"
        "僅 `apply_person_adjacent_exclusions`；**預設不移除**。"
        "敏感度診斷，最終採用待場地圖確認。\n"
    )
    md_lines.append("## 彙總表（prec / rec / P）\n")
    md_lines.append(
        "| 組 | 說明 | 0507 prec | 0507 rec | 0507 P | 0528 prec | 0528 rec | 0528 P | "
        "0528橋數 | 0507不變壞 | 0528橋消失 | 雙條件 |\n"
        "|----|------|-----------|----------|--------|-----------|----------|--------|"
        "---------|------------|------------|--------|\n"
    )
    for r in summary_rows:
        md_lines.append(
            f"| {r['group']} | {r['desc']} | "
            f"{r['0507_prec']:.3f} | {r['0507_rec']:.3f} | {r['0507_P']:.4f} | "
            f"{r['0528_prec']:.3f} | {r['0528_rec']:.3f} | {r['0528_P']:.4f} | "
            f"{r['0528_n_bridges_in_top1']} | "
            f"{'✓' if r['0507_not_worse'] else '✗紅旗'} | "
            f"{'✓' if r['0528_bridges_gone'] else '✗'} | "
            f"{'★' if r['meets_both'] else '—'} |\n"
        )
    meets = results["meets_0507_ok_and_0528_bridges_gone"]
    md_lines.append(
        f"\n**同時滿足「0507 不變壞 + 0528 橋消失」：{meets if meets else '（無）'}**\n"
    )

    for short in ("0507", "0528"):
        md_lines.append(f"\n## {short}\n")
        dep = results["by_dataset"][short]["gt_dependency_T0"]
        md_lines.append("### GT 對 focus 邊的依賴（全量拓撲 T0）\n")
        md_lines.append(
            f"- hop1 直連 focus：{len(dep['hop1_direct_on_focus'])} 條\n"
        )
        for e in dep["hop1_direct_on_focus"]:
            md_lines.append(
                f"  - `{e['from']}→{e['to']}` ({e['cams']}) dt={e['dt']:.2f} "
                f"**不能砍证据：{e['depends_on']}**\n"
            )
        md_lines.append(
            f"- hop2 使用 focus 邊：{len(dep['hop2_using_focus_edges'])} 條\n"
        )
        for e in dep["hop2_using_focus_edges"][:20]:
            md_lines.append(
                f"  - `{e['from']}→{e['to']}` hop2 via {e['common_neighbors']} "
                f"uses {e['uses_focus_edges']}\n"
            )
        if dep["cannot_cut_if_nonempty"]:
            md_lines.append("\n> **有 GT→GT 依賴 → 不能輕易砍（除非場地圖否認相鄰）。**\n")

        for tid in ABLATIONS:
            g = results["by_dataset"][short]["groups"][tid]
            md_lines.append(f"\n### {tid} — {g['desc']}\n")
            t = g["top1"]
            md_lines.append(
                f"- Top-1：prec={t['precision']:.3f} rec={t['recall']:.3f} "
                f"P={t['path_probability']:.4f}\n"
            )
            md_lines.append(f"- 路徑：`{t['path']}`\n")
            md_lines.append(f"- 圖：`{g['viz']}`\n")
            if g.get("bridges_in_top1"):
                md_lines.append(f"- 橋節點：`{g['bridges_in_top1']}`\n")
            if g.get("regression_0507"):
                rg = g["regression_0507"]
                flag = "紅旗" if rg["red_flag"] else "OK"
                md_lines.append(
                    f"- 0507 回歸 vs T0：{flag} "
                    f"(Δrec={rg['recall_delta']:+.3f} Δprec={rg['precision_delta']:+.3f} "
                    f"path_same={rg['path_same_as_T0']})\n"
                )
            ch = g["gt_edge_changes_vs_full"]
            md_lines.append(f"- 相對全量拓撲，GT 邊合法性/hop 變化：{len(ch)} 條\n")
            for c in ch[:15]:
                md_lines.append(
                    f"  - `{c['from']}→{c['to']}` t0_hop={c['t0_hop']} → "
                    f"ok={c['new_ok']} hop={c['new_hop']} {c['new_reason']}\n"
                )

    md_path = OUT / "topo_ablation_0509_0709.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")

    # strip nothing else; JSON without tracks
    out_json = OUT / "topo_ablation_0509_0709.json"
    out_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log(f"\n寫入 {out_json}")
    log(f"報告 {md_path}")
    log(f"視覺化 {VIZ}")
    log(f"雙條件組：{meets}")
    log(f"總耗時 {time.time()-t_all:.1f}s")

    # restore default exclusions
    pes.reset_person_adjacent_exclusions()
    return results


if __name__ == "__main__":
    main()
