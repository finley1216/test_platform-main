# -*- coding: utf-8 -*-
"""
任務一～四整合：橋帳目 / 共存盤查 / 自校準重跑 / prior 整合實驗
============================================================
GT 只用於評估與共存盤查，不進任何計分／校準計算。
B 設定：dt off, prior off（任務四 prior on 為變項）, EMB 0.80, supernode, node_evidence 依組。
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import llr_gate_config as gates  # noqa: E402
from evaluate_paths import load_gt  # noqa: E402
from run_b_exact_viz import score_labeled_path  # noqa: E402
import calibrate_self as cself  # noqa: E402


def precision_recall(path_tids: list[str], gt_set: set[str], n_gt: int) -> dict:
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    return {
        "n_path": n,
        "n_hit": hit,
        "precision": (hit / n) if n else 0.0,
        "recall": (hit / float(n_gt)) if n_gt else 0.0,
    }

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

OUT = OUTPUT_ROOT / "path_enum_llr"
CALIB_0507 = OUT / "calibration_gt0507.pkl"
CALIB_SELF = OUT / "calibration_self0528.pkl"

MERGE_0528 = QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528"
MERGE_0507 = QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"
GT_0528 = OUT / "ground_truth_20260528.json"
GT_0507 = OUT / "ground_truth_20260507.json"

BRIDGES = ["K8-07_1", "K8-07_93", "K8-09_96", "K8-07_139", "K8-09_167"]

# A = 現況 Top-1（含 09_96、07_139 橋）
PATH_A = [
    "K8-07_1",
    "K8-09_3",
    "{K8-08_17,K8-01_8}",
    "K8-05_10",
    "{K8-23_3,K8-22_26}",
    "K8-22_10",
    "K8-08_73",
    "K8-08_97",
    "K8-07_93",
    "K8-09_94",
    "K8-09_96",
    "K8-07_139",
    "{K8-09_142,K8-10_32}",
    "{K8-08_151,K8-01_58}",
    "K8-09_167",
]

# B 誠實分段
PATH_B_SEG1 = [
    "K8-07_1",
    "K8-09_3",
    "{K8-08_17,K8-01_8}",
    "K8-05_10",
    "{K8-23_3,K8-22_26}",
    "K8-22_10",
    "K8-08_73",
    "K8-08_97",
    "K8-07_93",
    "K8-09_94",
    "K8-12_14",
    "K8-30_5",
]
PATH_B_SEG2 = [
    "{K8-09_142,K8-10_32}",
    "{K8-08_151,K8-01_58}",
    "K8-09_167",
]

BRIDGE_NODES = {"K8-09_96", "K8-07_139"}
CORRIDOR_NODES = {"K8-12_14", "K8-30_5"}


def _path_str(labels):
    return " -> ".join(labels)


def _detail_breakdown(result: dict) -> dict:
    """拆節點分 + 邊分。"""
    nodes = []
    for ne in result.get("node_evidence") or []:
        nodes.append(
            {
                "super": ne.get("super"),
                "members": ne.get("members"),
                "sim": ne.get("sim"),
                "raw": ne.get("raw"),
                "w": ne.get("w"),
                "score": ne.get("score"),
            }
        )
    edges = []
    for e in result.get("edges") or []:
        edges.append(
            {
                "from_super": e.get("from_super"),
                "to_super": e.get("to_super"),
                "via": e.get("via"),
                "dt": e.get("dt"),
                "hop": e.get("hop"),
                "emb": e.get("emb"),
                "LLR_emb": e.get("LLR_emb"),
                "LLR_dH": e.get("LLR_dH"),
                "LLR_dt": e.get("LLR_dt"),
                "LLR_transition": e.get("LLR_transition"),
                "score": e.get("score"),
                "dt_model": e.get("dt_model"),
            }
        )
    return {
        "ok": result.get("ok"),
        "reason": result.get("reason"),
        "score": result.get("score"),
        "super_labels": result.get("super_labels"),
        "nodes": nodes,
        "edges": edges,
        "node_sum": float(sum(n["score"] or 0 for n in nodes)),
        "edge_sum": float(sum(e["score"] or 0 for e in edges)),
    }


def _label_set(lab: str) -> set[str]:
    if lab.startswith("{") and lab.endswith("}"):
        return {x.strip() for x in lab[1:-1].split(",") if x.strip()}
    return {lab}


def _involves(lab: str, focus: set[str]) -> bool:
    return bool(_label_set(lab) & focus)


def task1_bridge_vs_segment(tracks, nodes, calib) -> dict:
    """精確計分（不經 beam）比較 A vs B。"""
    a = score_labeled_path(nodes, PATH_A, calib, dt_scoring=False, transition_prior=False)
    b1 = score_labeled_path(
        nodes, PATH_B_SEG1, calib, dt_scoring=False, transition_prior=False
    )
    b2 = score_labeled_path(
        nodes, PATH_B_SEG2, calib, dt_scoring=False, transition_prior=False
    )
    da, db1, db2 = _detail_breakdown(a), _detail_breakdown(b1), _detail_breakdown(b2)
    b_total = None
    if db1["ok"] and db2["ok"]:
        b_total = float(db1["score"]) + float(db2["score"])

    # A 橋貢獻：節點 09_96 / 07_139 + 進出邊（任一端為橋）
    bridge_node_scores = []
    for n in da.get("nodes") or []:
        mems = set(n.get("members") or [])
        if mems & BRIDGE_NODES or _involves(n.get("super") or "", BRIDGE_NODES):
            bridge_node_scores.append(n)
    bridge_edge_scores = []
    for e in da.get("edges") or []:
        if _involves(e.get("from_super") or "", BRIDGE_NODES) or _involves(
            e.get("to_super") or "", BRIDGE_NODES
        ):
            bridge_edge_scores.append(e)

    # B 走廊貢獻
    corridor_node_scores = []
    for n in db1.get("nodes") or []:
        mems = set(n.get("members") or [])
        if mems & CORRIDOR_NODES or _involves(n.get("super") or "", CORRIDOR_NODES):
            corridor_node_scores.append(n)
    corridor_edge_scores = []
    for e in db1.get("edges") or []:
        if _involves(e.get("from_super") or "", CORRIDOR_NODES) or _involves(
            e.get("to_super") or "", CORRIDOR_NODES
        ):
            corridor_edge_scores.append(e)

    # 共享前綴至 09_94（不含之後）
    prefix = PATH_A[:10]  # … → 09_94
    pref = score_labeled_path(
        nodes, prefix, calib, dt_scoring=False, transition_prior=False
    )
    dpref = _detail_breakdown(pref)

    # A 後綴：09_94 → 橋 → 晚段
    a_suffix_labels = PATH_A[9:]  # from 09_94
    a_suf = score_labeled_path(
        nodes, a_suffix_labels, calib, dt_scoring=False, transition_prior=False
    )
    da_suf = _detail_breakdown(a_suf)

    # B 後綴走廊：09_94 → 12 → 30
    b_suf_labels = PATH_B_SEG1[9:]
    b_suf = score_labeled_path(
        nodes, b_suf_labels, calib, dt_scoring=False, transition_prior=False
    )
    db_suf = _detail_breakdown(b_suf)

    bridge_node_sum = sum(n["score"] or 0 for n in bridge_node_scores)
    bridge_edge_sum = sum(e["score"] or 0 for e in bridge_edge_scores)
    corridor_node_sum = sum(n["score"] or 0 for n in corridor_node_scores)
    corridor_edge_sum = sum(e["score"] or 0 for e in corridor_edge_scores)

    # 注意：路徑總分 ≠ 後綴加總（節點證據在路徑裡只算一次；後綴重算會重算 09_94）
    # 帳目以「差分區塊」呈現：
    # A_extra_from_bridge ≈ A_suffix_nodes/edges touching bridges + late chain
    # B_corridor_earn vs A late continuity

    delta = None
    if da["ok"] and b_total is not None:
        delta = float(da["score"]) - float(b_total)

    return {
        "settings": "B: dt=off prior=off emb_gate supernode node_evidence；精確標註路徑計分",
        "path_A": _path_str(PATH_A),
        "path_B_seg1": _path_str(PATH_B_SEG1),
        "path_B_seg2": _path_str(PATH_B_SEG2),
        "A": da,
        "B_seg1": db1,
        "B_seg2": db2,
        "B_total_score": b_total,
        "delta_A_minus_B": delta,
        "shared_prefix_to_09_94": dpref,
        "A_suffix_from_09_94": da_suf,
        "B_corridor_suffix_from_09_94": db_suf,
        "bridge_accounting": {
            "nodes": bridge_node_scores,
            "edges": bridge_edge_scores,
            "node_sum": bridge_node_sum,
            "edge_sum": bridge_edge_sum,
            "total": bridge_node_sum + bridge_edge_sum,
            "note": "A 路徑上 09_96/07_139 節點分 + 任一端為橋的邊分（含進出走晚段的邊）",
        },
        "corridor_accounting": {
            "nodes": corridor_node_scores,
            "edges": corridor_edge_scores,
            "node_sum": corridor_node_sum,
            "edge_sum": corridor_edge_sum,
            "total": corridor_node_sum + corridor_edge_sum,
            "note": "B seg1 上 12_14/30_5 節點分 + 走廊邊分（死路區間賺到的部分）",
        },
        "interpretation": {
            "A_bridge_block_total": bridge_node_sum + bridge_edge_sum,
            "B_corridor_block_total": corridor_node_sum + corridor_edge_sum,
            "B_seg2_independent": db2.get("score"),
            "A_includes_late_chain_in_one_path": True,
            "B_seg2_node_sum": db2.get("node_sum"),
            "B_seg2_edge_sum": db2.get("edge_sum"),
        },
    }


def task2_coexistence_audit(tracks, gt_tids: list[str]) -> dict:
    by_tid = {t.tid: t for t in tracks}
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    missing_gt = [t for t in gt_tids if t not in by_tid]
    missing_bridge = [t for t in BRIDGES if t not in by_tid]

    rows = []
    contradictions = []
    special = []

    for btid in BRIDGES:
        bt = by_tid.get(btid)
        if bt is None:
            rows.append({"bridge": btid, "error": "not_in_candidate_pool"})
            continue
        for gt in gt_tracks:
            if gt.tid == btid:
                continue
            ov = max(
                0.0,
                min(bt.t_end, gt.t_end) - max(bt.t_start, gt.t_start),
            )
            # 點重疊
            if ov <= 0:
                tok, ov2, _ = llr._coexistence_time_ok(bt, gt)
                if not tok:
                    continue
                ov = ov2 if ov2 > 0 else 1e-6

            key = tuple(sorted((bt.cam, gt.cam)))
            is_ov = key in pes.OVERLAP_PAIRS
            is_adj = key in pes.ADJACENT
            same_cam = bt.cam == gt.cam
            contradiction = (not same_cam) and (not is_ov) and (not is_adj)
            row = {
                "bridge": btid,
                "gt": gt.tid,
                "bridge_cam": bt.cam,
                "gt_cam": gt.cam,
                "overlap_sec": float(ov),
                "same_cam": same_cam,
                "OVERLAP": is_ov,
                "ADJACENT": is_adj,
                "coexistence_contradiction": contradiction,
            }
            rows.append(row)
            if contradiction:
                contradictions.append(row)
            if btid in ("K8-09_96", "K8-07_139") and gt.tid in (
                "K8-12_14",
                "K8-30_5",
            ):
                special.append(row)

    return {
        "bridges": BRIDGES,
        "n_gt": len(gt_tids),
        "missing_gt": missing_gt,
        "missing_bridge": missing_bridge,
        "n_overlap_pairs_checked": len(rows),
        "n_contradictions": len(contradictions),
        "contradictions": contradictions,
        "special_09_96_07_139_vs_12_14_30_5": special,
        "note": "矛盾＝時間重疊且跨鏡且非 OVERLAP 非 ADJACENT；只記錄不改演算法。GT 僅供盤查。",
    }


def merge_self_with_node_ev_0507(self_calib: dict, gt_calib: dict) -> dict:
    c = deepcopy(self_calib)
    c["sim_gt"] = deepcopy(gt_calib["sim_gt"])
    c["sim_nongt"] = deepcopy(gt_calib["sim_nongt"])
    c.setdefault("meta", {})["node_evidence_source"] = "calibration_gt0507 sim_*"
    return c


def bridges_in_path(tids_or_labels) -> dict:
    flat = set()
    for x in tids_or_labels or []:
        if isinstance(x, str) and x.startswith("{"):
            flat |= _label_set(x)
        else:
            flat.add(x)
    return {b: (b in flat) for b in BRIDGES}


def run_one(
    merge_dir: Path,
    calib: dict,
    gt_path: Path,
    *,
    use_node_evidence: bool,
    transition_prior: bool,
    tag: str,
) -> dict:
    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge_dir))
    gt = load_gt(gt_path)
    gt_tids = list(gt["person_tids"])
    gt_set = set(gt_tids)
    n_gt = len(gt_tids)
    (
        _tracks,
        scored,
        maximal,
        n_legal_edges,
        _nodes,
        _super_report,
        _gate_info,
        _options,
    ) = llr.run_llr(
        merge_dir,
        calib,
        use_emb_gate_fix=True,
        use_supernode=True,
        use_node_evidence=use_node_evidence,
        dt_scoring=False,
        transition_prior=transition_prior,
    )
    top = (maximal or [None])[0]
    if not top:
        return {"tag": tag, "ok": False}
    path_labs = top.get("super_labels") or top.get("tids")
    pr = precision_recall(top["tids"], gt_set, n_gt)
    br = bridges_in_path(top.get("tids") or [])
    return {
        "tag": tag,
        "ok": True,
        "prec": pr["precision"],
        "rec": pr["recall"],
        "hit": pr["n_hit"],
        "n_path": pr["n_path"],
        "n_gt": n_gt,
        "P": top.get("path_probability"),
        "score": top.get("score"),
        "path": " -> ".join(path_labs),
        "bridges_in_top1": br,
        "n_bridges": sum(1 for v in br.values() if v),
        "n_legal_edges": n_legal_edges,
        "n_paths": len(scored),
        "use_node_evidence": use_node_evidence,
        "transition_prior": transition_prior,
        "emb_same_n": (calib.get("emb_same") or {}).get("n"),
        "emb_w": (calib.get("emb_same") or {}).get("shrink_w")
        or (
            float((calib.get("emb_same") or {}).get("n") or 0)
            / (float((calib.get("emb_same") or {}).get("n") or 0) + 10.0)
        ),
        "p_edge": (calib.get("transition_prior") or {}).get("p_edge"),
    }


def write_markdown(report: dict, out_md: Path) -> None:
    t1 = report["task1"]
    t2 = report["task2"]
    t3 = report["task3"]
    t4 = report.get("task4")

    lines = [
        "# 橋帳目 / 共存盤查 / 自校準 / prior 整合",
        "",
        "> GT 只用於評估與共存盤查，不進計分／自校準。",
        "> 凍結 B：dt off；EMB 0.80；supernode。",
        "",
        "## 任務一：橋 vs 分段精確帳目（0528，B，calibration_gt0507）",
        "",
        f"- **A score** = `{t1['A'].get('score')}`  "
        f"(node={t1['A'].get('node_sum'):.4f} edge={t1['A'].get('edge_sum'):.4f})",
        f"- **B total** = seg1 `{t1['B_seg1'].get('score')}` + seg2 `{t1['B_seg2'].get('score')}` "
        f"= `{t1.get('B_total_score')}`",
        f"- **Δ(A−B)** = `{t1.get('delta_A_minus_B')}`",
        "",
        f"- A 路徑：`{t1['path_A']}`",
        f"- B seg1：`{t1['path_B_seg1']}`",
        f"- B seg2：`{t1['path_B_seg2']}`",
        "",
        "### A 因橋多賺的區塊（09_96 / 07_139）",
        f"- 節點合計 **{t1['bridge_accounting']['node_sum']:.4f}**",
    ]
    for n in t1["bridge_accounting"]["nodes"]:
        lines.append(
            f"  - node `{n['super']}` score={n['score']:.4f} "
            f"(sim={n.get('sim'):.4f} raw={n.get('raw'):.4f} w={n.get('w'):.4f})"
        )
    lines.append(f"- 邊合計 **{t1['bridge_accounting']['edge_sum']:.4f}**")
    for e in t1["bridge_accounting"]["edges"]:
        lines.append(
            f"  - edge `{e['from_super']}→{e['to_super']}` via {e['via']} "
            f"dt={e['dt']:.2f} hop={e['hop']} emb={e['emb']:.4f} "
            f"LLR_emb={e['LLR_emb']:.4f} score={e['score']:.4f}"
        )
    lines.append(
        f"- **橋區塊合計 {t1['bridge_accounting']['total']:.4f}**"
    )

    lines += [
        "",
        "### B 因走廊（12_14 / 30_5）賺到的部分（死路／誠實收尾）",
        f"- 節點合計 **{t1['corridor_accounting']['node_sum']:.4f}**",
    ]
    for n in t1["corridor_accounting"]["nodes"]:
        lines.append(
            f"  - node `{n['super']}` score={n['score']:.4f} "
            f"(sim={n.get('sim'):.4f})"
        )
    lines.append(f"- 邊合計 **{t1['corridor_accounting']['edge_sum']:.4f}**")
    for e in t1["corridor_accounting"]["edges"]:
        lines.append(
            f"  - edge `{e['from_super']}→{e['to_super']}` via {e['via']} "
            f"dt={e['dt']:.2f} hop={e['hop']} emb={e['emb']:.4f} "
            f"LLR_emb={e['LLR_emb']:.4f} score={e['score']:.4f}"
        )
    lines.append(
        f"- **走廊區塊合計 {t1['corridor_accounting']['total']:.4f}**"
    )
    lines += [
        "",
        "### 對照",
        f"- 共享前綴至 09_94 score=`{t1['shared_prefix_to_09_94'].get('score')}` "
        f"ok={t1['shared_prefix_to_09_94'].get('ok')}",
        f"- A 自 09_94 後綴 score=`{t1['A_suffix_from_09_94'].get('score')}` "
        f"ok={t1['A_suffix_from_09_94'].get('ok')}",
        f"- B 走廊後綴 score=`{t1['B_corridor_suffix_from_09_94'].get('score')}` "
        f"ok={t1['B_corridor_suffix_from_09_94'].get('ok')}",
        f"- B seg2（獨立）score=`{t1['B_seg2'].get('score')}` "
        f"node={t1['B_seg2'].get('node_sum'):.4f} edge={t1['B_seg2'].get('edge_sum'):.4f}",
    ]
    if not t1["B_seg1"].get("ok"):
        lines.append(f"- ⚠ B seg1 失敗：{t1['B_seg1'].get('reason')}")
    if not t1["B_seg2"].get("ok"):
        lines.append(f"- ⚠ B seg2 失敗：{t1['B_seg2'].get('reason')}")

    lines += [
        "",
        "## 任務二：橋節點共存盤查（只記錄）",
        f"- 矛盾數：**{t2['n_contradictions']}**",
        "",
    ]
    if t2["contradictions"]:
        lines.append("| bridge | GT | cams | overlap_s | OVERLAP | ADJACENT |")
        lines.append("|--------|----|------|-----------|---------|----------|")
        for r in t2["contradictions"]:
            lines.append(
                f"| {r['bridge']} | {r['gt']} | {r['bridge_cam']}↔{r['gt_cam']} | "
                f"{r['overlap_sec']:.2f} | {r['OVERLAP']} | {r['ADJACENT']} |"
            )
    else:
        lines.append("（無矛盾）")

    lines += ["", "### 特別：09_96 / 07_139 vs 12_14 / 30_5", ""]
    if t2["special_09_96_07_139_vs_12_14_30_5"]:
        lines.append("| bridge | GT | cams | overlap_s | 矛盾 |")
        lines.append("|--------|----|------|-----------|------|")
        for r in t2["special_09_96_07_139_vs_12_14_30_5"]:
            lines.append(
                f"| {r['bridge']} | {r['gt']} | {r['bridge_cam']}↔{r['gt_cam']} | "
                f"{r['overlap_sec']:.2f} | {r['coexistence_contradiction']} |"
            )
    else:
        lines.append("（無時間重疊列）")

    sc = t3["self_calib_summary"]
    lines += [
        "",
        "## 任務三：自校準 + 重跑",
        "",
        "### 分布對照",
        f"- self emb|same：n={sc['self']['emb_same']['n']} "
        f"μ={sc['self']['emb_same']['mu']:.4f} σ={sc['self']['emb_same']['sigma']:.4f} "
        f"w={sc['self']['emb_same']['shrink_w']:.4f}",
        f"- self emb|diff：n={sc['self']['emb_diff']['n']} "
        f"μ={sc['self']['emb_diff']['mu']:.4f} σ={sc['self']['emb_diff']['sigma']:.4f} "
        f"w={sc['self']['emb_diff']['shrink_w']:.4f}",
        f"- 0507 emb|same：n={sc['gt0507']['emb_same']['n']} "
        f"μ={sc['gt0507']['emb_same']['mu']:.4f} σ={sc['gt0507']['emb_same']['sigma']:.4f} "
        f"w={sc['gt0507']['emb_same']['shrink_w']:.4f}",
        f"- 0507 emb|diff：n={sc['gt0507']['emb_diff']['n']} "
        f"μ={sc['gt0507']['emb_diff']['mu']:.4f} σ={sc['gt0507']['emb_diff']['sigma']:.4f} "
        f"w={sc['gt0507']['emb_diff']['shrink_w']:.4f}",
        f"- **emb w 是否接近 1**：self_w={sc['self']['emb_same']['shrink_w']:.4f} "
        f"（閾值用 ≥0.9 → {sc['emb_w_near_one']}）",
        "",
        "### 0528 Top-1",
        "",
        "| 組 | prec | rec | P | 橋數 | 路徑 |",
        "|----|------|-----|---|------|------|",
    ]
    for key in ["baseline_gt0507", "self_node_ev_0507", "self_node_ev_off"]:
        r = t3["runs"][key]
        lines.append(
            f"| {key} | {r['prec']:.3f} | {r['rec']:.3f} | {r['P']:.4f} | "
            f"{r['n_bridges']}/5 | `{r['path']}` |"
        )
        br = r["bridges_in_top1"]
        lines.append(
            f"|  └ bridges | "
            + ", ".join(f"{k.split('_')[-1]}={'Y' if v else 'N'}" for k, v in br.items())
            + " |||| |"
        )

    if t4:
        lines += [
            "",
            "## 任務四：自校準 + transition-prior",
            f"- 觸發條件 emb w≥0.9：{t4.get('triggered')}",
            f"- self p_edge={t4.get('self_p_edge')}（NO GT，多成員超節點相鄰合法/全圖合法邊）",
            "",
            "### 0528",
            "| 組 | prec | rec | P | 橋 |",
            "|----|------|-----|---|----|",
        ]
        for key in ["self_prior_off", "self_prior_on"]:
            r = t4["runs_0528"][key]
            lines.append(
                f"| {key} | {r['prec']:.3f} | {r['rec']:.3f} | {r['P']:.4f} | "
                f"{r['n_bridges']}/5 |"
            )
            lines.append(f"|  └ path | `{r['path']}` ||||")
        lines += [
            "",
            "### 0507 回歸（等效旗標；calibration_gt0507）",
            "| 組 | prec | rec | P | path_same_as_B_baseline |",
            "|----|------|-----|---|-------------------------|",
        ]
        for key, r in t4["runs_0507"].items():
            lines.append(
                f"| {key} | {r['prec']:.3f} | {r['rec']:.3f} | {r['P']:.4f} | "
                f"{r.get('path_same_as_baseline')} |"
            )
    else:
        lines += [
            "",
            "## 任務四：略過",
            f"- 原因：{report.get('task4_skip_reason')}",
        ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pes.SIM_MIN = 0.85  # 與 B／現行 0528 實驗一致

    # --- 自校準 ---
    print("======== calibrate_self 0528 ========")
    pes.configure_for_input(str(MERGE_0528))
    tracks_0528 = pes.load_tracks(str(MERGE_0528))
    print(f"載入 tracks={len(tracks_0528)}  SIM_MIN={pes.SIM_MIN}")
    gates.apply_llr_emb_gates(True)
    gt0507_calib = pickle.loads(CALIB_0507.read_bytes())
    samples = cself.collect_self_samples(tracks_0528)
    self_calib = cself.fit_self_calibration(samples, gt0507_calib)
    self_prior = cself.compute_self_transition_prior(tracks_0528)
    self_calib["transition_prior"] = self_prior
    CALIB_SELF.write_bytes(pickle.dumps(self_calib))
    cself.write_report(
        self_calib,
        samples,
        self_prior,
        gt0507_calib,
        OUT / "calibration_self0528_report.txt",
    )
    cself.save_hist(
        samples["emb_same"],
        samples["emb_diff"],
        OUT / "emb_same_diff_hist_self0528.png",
    )
    print(f"寫入 {CALIB_SELF}")

    emb_w = float(self_calib["emb_same"]["shrink_w"])
    emb_w_near_one = emb_w >= 0.9

    # --- 任務一 ---
    print("======== 任務一：橋 vs 分段 ========")
    gates.apply_llr_emb_gates(True)
    nodes_0528, _ = llr.build_supernodes(tracks_0528)
    task1 = task1_bridge_vs_segment(tracks_0528, nodes_0528, gt0507_calib)
    print(
        f"A={task1['A'].get('score')}  B={task1.get('B_total_score')}  "
        f"Δ={task1.get('delta_A_minus_B')}"
    )

    # --- 任務二 ---
    print("======== 任務二：共存盤查 ========")
    gt_0528 = list(load_gt(GT_0528)["person_tids"])
    task2 = task2_coexistence_audit(tracks_0528, gt_0528)
    print(f"矛盾 {task2['n_contradictions']} 條；特別列 {len(task2['special_09_96_07_139_vs_12_14_30_5'])}")

    # --- 任務三 ---
    print("======== 任務三：自校準重跑 ========")
    self_ne0507 = merge_self_with_node_ev_0507(self_calib, gt0507_calib)
    runs3 = {
        "baseline_gt0507": run_one(
            MERGE_0528,
            gt0507_calib,
            GT_0528,
            use_node_evidence=True,
            transition_prior=False,
            tag="baseline_gt0507",
        ),
        "self_node_ev_0507": run_one(
            MERGE_0528,
            self_ne0507,
            GT_0528,
            use_node_evidence=True,
            transition_prior=False,
            tag="self_node_ev_0507",
        ),
        "self_node_ev_off": run_one(
            MERGE_0528,
            self_calib,
            GT_0528,
            use_node_evidence=False,
            transition_prior=False,
            tag="self_node_ev_off",
        ),
    }
    task3 = {
        "self_calib_summary": {
            "self": {
                "emb_same": self_calib["emb_same"],
                "emb_diff": self_calib["emb_diff"],
            },
            "gt0507": {
                "emb_same": gt0507_calib["emb_same"],
                "emb_diff": gt0507_calib["emb_diff"],
            },
            "sample_counts": samples["counts"],
            "emb_w_near_one": emb_w_near_one,
            "self_transition_prior": self_prior,
        },
        "runs": runs3,
    }

    # --- 任務四 ---
    task4 = None
    task4_skip = None
    if emb_w_near_one:
        print("======== 任務四：prior 整合 ========")
        # 對照：self + prior off（沿用 node_ev 0507，與任務三可比）
        # 實驗：self + prior on（p_edge=self）
        calib_prior_off = merge_self_with_node_ev_0507(self_calib, gt0507_calib)
        calib_prior_on = deepcopy(calib_prior_off)
        calib_prior_on["transition_prior"] = self_prior

        runs4_0528 = {
            "self_prior_off": run_one(
                MERGE_0528,
                calib_prior_off,
                GT_0528,
                use_node_evidence=True,
                transition_prior=False,
                tag="self_prior_off",
            ),
            "self_prior_on": run_one(
                MERGE_0528,
                calib_prior_on,
                GT_0528,
                use_node_evidence=True,
                transition_prior=True,
                tag="self_prior_on",
            ),
        }
        # 0507 回歸：等效旗標 + gt0507 calib
        base_0507 = run_one(
            MERGE_0507,
            gt0507_calib,
            GT_0507,
            use_node_evidence=True,
            transition_prior=False,
            tag="0507_B_baseline",
        )
        prior_on_0507 = run_one(
            MERGE_0507,
            gt0507_calib,
            GT_0507,
            use_node_evidence=True,
            transition_prior=True,
            tag="0507_B_prior_on",
        )
        prior_on_0507["path_same_as_baseline"] = prior_on_0507.get("path") == base_0507.get(
            "path"
        )
        base_0507["path_same_as_baseline"] = True
        task4 = {
            "triggered": True,
            "self_p_edge": self_prior.get("p_edge"),
            "self_ln_p_edge": self_prior.get("ln_p_edge"),
            "runs_0528": runs4_0528,
            "runs_0507": {
                "B_prior_off": base_0507,
                "B_prior_on": prior_on_0507,
            },
        }
    else:
        task4_skip = (
            f"emb|same shrink_w={emb_w:.4f} < 0.9，未觸發 prior 整合實驗"
        )
        print(task4_skip)

    report = {
        "warning": "GT 只用於評估與共存盤查；自校準／計分不碰 GT。",
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4,
        "task4_skip_reason": task4_skip,
        "elapsed_sec": time.time() - t0,
    }

    # JSON：精簡過大的 node/edge 列表已足夠
    json_path = OUT / "bridge_selfcalib_prior_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path = OUT / "bridge_selfcalib_prior_report.md"
    write_markdown(report, md_path)
    print(f"寫入 {json_path}")
    print(f"寫入 {md_path}")
    print(f"總耗時 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
