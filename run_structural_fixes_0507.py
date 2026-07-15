# -*- coding: utf-8 -*-
"""
結構修正重跑：超節點可行性 + 三套對照 + 消融 + 08_43 Top-10 檢查
================================================================
不動 path_enum_scoring.py。結果寫入 comparison_gt_20260507.md 與 JSON。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import llr_gate_config as gates  # noqa: E402
import path_enum_llr as llr  # noqa: E402
from evaluate_paths import (  # noqa: E402
    N_GT,
    load_gt,
    precision_recall,
    find_gt_best_path,
    rank_of_path,
)

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

EXPECTED_CHAIN_LABELS = [
    "K8-09_7",
    "{K8-08_30,K8-01_7}",
    "K8-07_40",
    "{K8-23_8,K8-22_19}",
    "K8-22_22",
    "K8-07_112",
    "K8-01_50",
    "{K8-08_77,K8-01_62}",
]
EXPECTED_EXPAND = [
    "K8-09_7",
    "K8-08_30",
    "K8-01_7",
    "K8-07_40",
    "K8-23_8",
    "K8-22_19",
    "K8-22_22",
    "K8-07_112",
    "K8-01_50",
    "K8-08_77",
    "K8-01_62",
]


def hop_info(cam_a: str, cam_b: str) -> dict:
    key = tuple(sorted((cam_a, cam_b)))
    hop = pes.hop_count(cam_a, cam_b)
    return {
        "pair": f"{cam_a}↔{cam_b}",
        "in_ADJACENT": key in pes.ADJACENT,
        "hop_count": hop,
        "adjacent_hop1": hop == 1,
    }


def diagnose_super_chain(tracks: list, expected_member_groups: list[list[str]]) -> dict:
    """檢查期望超節點鏈是否邊邊通過（修正一門檻已套用）。"""
    by_tid = {t.tid: t for t in tracks}
    # 建實際超節點並對齊期望
    supers, srep = llr.build_supernodes(tracks)
    by_member = {}
    for sn in supers:
        for tid in sn.tids:
            by_member[tid] = sn

    chain_nodes = []
    chain_ok = True
    detail = []
    for group in expected_member_groups:
        sns = {by_member[t].sid for t in group if t in by_member}
        if len(sns) != 1:
            chain_ok = False
            detail.append(
                {
                    "expected_members": group,
                    "ok": False,
                    "reason": f"未合成同一超節點（sids={sns}）",
                }
            )
            chain_nodes.append(None)
            continue
        sn = by_member[group[0]]
        if set(sn.tids) != set(group) and not set(group).issubset(set(sn.tids)):
            # 允許單例；多成員必須涵蓋期望
            pass
        if len(group) > 1 and set(group) != set(sn.tids):
            detail.append(
                {
                    "expected_members": group,
                    "actual_members": sn.tids,
                    "ok": set(group) == set(sn.tids),
                    "note": "成員不完全一致" if set(group) != set(sn.tids) else "ok",
                }
            )
        chain_nodes.append(sn)

    edge_checks = []
    all_edges_ok = True
    for i in range(len(chain_nodes) - 1):
        sa, sb = chain_nodes[i], chain_nodes[i + 1]
        if sa is None or sb is None:
            all_edges_ok = False
            edge_checks.append(
                {
                    "from": expected_member_groups[i],
                    "to": expected_member_groups[i + 1],
                    "ok": False,
                    "reason": "超節點未形成",
                }
            )
            continue
        best, rejects = llr._best_member_edge(sa, sb)
        if best is None:
            all_edges_ok = False
            # 彙總拒絕原因（取最具代表性）
            reasons = {}
            for a, b, r in rejects:
                reasons[r] = reasons.get(r, 0) + 1
            # 也對期望代表做一次 edge_check 解釋
            u0 = by_tid.get(expected_member_groups[i][0])
            v0 = by_tid.get(expected_member_groups[i + 1][0])
            expl = None
            if u0 and v0:
                ok, reason, dt, hop, emb, h_dist = pes.edge_check(u0, v0)
                expl = {
                    "pair": f"{u0.tid}->{v0.tid}",
                    "ok": ok,
                    "reason": reason,
                    "dt": dt,
                    "hop": hop,
                    "emb": emb,
                    "h_dist": h_dist,
                }
            edge_checks.append(
                {
                    "from": sa.label,
                    "to": sb.label,
                    "ok": False,
                    "reject_counts": reasons,
                    "sample_reject": rejects[:5],
                    "example_edge_check": expl,
                }
            )
        else:
            u, v, dt, hop, emb, h_dist = best
            # hist gate along the chain prefix?
            edge_checks.append(
                {
                    "from": sa.label,
                    "to": sb.label,
                    "ok": True,
                    "via": f"{u.tid}->{v.tid}",
                    "dt": float(dt),
                    "hop": hop,
                    "emb": float(emb),
                    "h_dist": float(h_dist) if h_dist is not None else None,
                }
            )

    # 沿期望鏈做 EMB_HIST 檢查（超節點平均 emb）
    hist_ok = True
    hist_fails = []
    if all(x is not None for x in chain_nodes):
        hist = [chain_nodes[0].emb]
        for i in range(1, len(chain_nodes)):
            class _P:
                pass
            p = _P()
            p.emb = chain_nodes[i].emb
            hsim = pes.hist_emb_sim(hist, p)
            need = pes.EMB_HIST_MIN
            # 找對應邊的 h_dist
            ed = edge_checks[i - 1]
            if ed.get("ok") and ed.get("h_dist") is not None and ed["h_dist"] < pes.H_DIST_GATE:
                need = pes.EMB_HIST_MIN - 0.02
            if hsim < need:
                hist_ok = False
                hist_fails.append(
                    {
                        "to": chain_nodes[i].label,
                        "hist_emb": hsim,
                        "need": need,
                    }
                )
            hist.append(chain_nodes[i].emb)

    cover = len(EXPECTED_EXPAND) if (all_edges_ok and hist_ok and chain_ok) else None
    return {
        "expected_labels": EXPECTED_CHAIN_LABELS,
        "expected_expand": EXPECTED_EXPAND,
        "supernode_formation": detail,
        "edge_checks": edge_checks,
        "hist_checks": {"ok": hist_ok, "fails": hist_fails},
        "chain_fully_feasible": bool(chain_ok and all_edges_ok and hist_ok),
        "single_path_cover_upper": cover if cover else _max_prefix_cover(edge_checks, hist_fails),
        "multi_only_actual": srep.get("multi_only"),
        "all_supernodes": srep.get("supernodes"),
    }


def _max_prefix_cover(edge_checks, hist_fails) -> int:
    """沿期望鏈能走到多遠（展開 tid 數）。"""
    n_nodes = len(EXPECTED_CHAIN_LABELS)
    reachable = 1
    for i, ed in enumerate(edge_checks):
        if not ed.get("ok"):
            break
        # hist fail at to-index i+1
        if any(f.get("to") == EXPECTED_CHAIN_LABELS[i + 1] for f in hist_fails):
            break
        reachable += 1
    # expand count for reachable supers
    groups = [
        ["K8-09_7"],
        ["K8-08_30", "K8-01_7"],
        ["K8-07_40"],
        ["K8-23_8", "K8-22_19"],
        ["K8-22_22"],
        ["K8-07_112"],
        ["K8-01_50"],
        ["K8-08_77", "K8-01_62"],
    ]
    return sum(len(g) for g in groups[:reachable])


def topk_metrics(maximal: list, gt_set: set, k: int = 3) -> list:
    out = []
    for i, p in enumerate(maximal[:k], 1):
        pr = precision_recall(p["tids"], gt_set)
        out.append(
            {
                "rank": i,
                "precision": pr["precision"],
                "recall": pr["recall"],
                "n_hit": pr["n_hit"],
                "n_path": pr["n_path"],
                "score": p["score"],
                "path_probability": p.get("path_probability"),
                "tids": p["tids"],
                "super_labels": p.get("super_labels"),
                "path": " -> ".join(p.get("super_labels") or p["tids"]),
            }
        )
    return out


def find_08_43_in_top10(maximal: list) -> dict:
    tid = "K8-08_43"
    hits = []
    for i, p in enumerate(maximal[:10], 1):
        if tid in p["tids"]:
            # 找出進出邊與節點證據
            node_ev = None
            for ne in p.get("node_evidence") or []:
                if tid in (ne.get("members") or []) or tid in str(ne.get("super") or ""):
                    node_ev = ne
                    break
            in_edges = []
            out_edges = []
            for e in p.get("edges") or []:
                members_from = e.get("from_members") or [e.get("from")]
                members_to = e.get("to_members") or [e.get("to")]
                if tid in members_to or e.get("to") == tid:
                    in_edges.append(e)
                if tid in members_from or e.get("from") == tid:
                    out_edges.append(e)
            hits.append(
                {
                    "rank": i,
                    "tids": p["tids"],
                    "super_labels": p.get("super_labels"),
                    "score": p["score"],
                    "path_probability": p.get("path_probability"),
                    "node_evidence_08_43": node_ev,
                    "in_edges": in_edges,
                    "out_edges": out_edges,
                    "all_edges": p.get("edges"),
                    "all_node_evidence": p.get("node_evidence"),
                }
            )
    return {
        "present_in_top10": bool(hits),
        "hits": hits,
        "verdict": "出現" if hits else "未出現",
    }


def run_once(merge_dir: Path, calib: dict, **opts):
    return llr.run_llr(merge_dir, calib, **opts)


def main():
    merge_dir = (QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507").resolve()
    out_root = (OUTPUT_ROOT / "path_enum_llr").resolve()
    gt_path = out_root / "ground_truth_20260507.json"
    calib_path = out_root / "calibration_gt0507.pkl"

    gt = load_gt(gt_path)
    gt_set = set(gt["person_tids"])
    gt_list = list(gt["person_tids"])

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge_dir))
    calib = llr.load_calibration(calib_path)

    # --- topology answers ---
    topo = {
        "K8-22↔K8-07": hop_info("K8-22", "K8-07"),
        "K8-07↔K8-01": hop_info("K8-07", "K8-01"),
        "K8-23↔K8-08": hop_info("K8-23", "K8-08"),
    }

    # --- 修正一門檻下超節點可行性 ---
    gates.apply_llr_emb_gates(True)
    tracks = pes.load_tracks(str(merge_dir))
    expected_groups = [
        ["K8-09_7"],
        ["K8-08_30", "K8-01_7"],
        ["K8-07_40"],
        ["K8-23_8", "K8-22_19"],
        ["K8-22_22"],
        ["K8-07_112"],
        ["K8-01_50"],
        ["K8-08_77", "K8-01_62"],
    ]
    feas = diagnose_super_chain(tracks, expected_groups)
    supers, srep = llr.build_supernodes(tracks)
    mislabel_check = llr.verify_mislabel_not_in_gt_super(srep, gt_set)

    # --- 舊法 Top-3（讀既有 JSON 若存在，否則重跑）---
    old_json = OUTPUT_ROOT / "path_enum" / "人員追蹤_20260507_top1.json"
    old_top = []
    if old_json.is_file():
        old = json.loads(old_json.read_text(encoding="utf-8"))
        # try top10 or reconstruct from paths if any
        for i, p in enumerate((old.get("top10_paths") or [])[:3], 1):
            tids = p.get("tids") or []
            pr = precision_recall(tids, gt_set)
            old_top.append(
                {
                    "rank": i,
                    "precision": pr["precision"],
                    "recall": pr["recall"],
                    "score": p.get("score"),
                    "path_probability": None,
                    "tids": tids,
                    "path": " -> ".join(tids),
                }
            )
        if not old_top and old.get("top1"):
            tids = old["top1"]["tids"]
            pr = precision_recall(tids, gt_set)
            old_top = [
                {
                    "rank": 1,
                    "precision": pr["precision"],
                    "recall": pr["recall"],
                    "score": old["top1"].get("score"),
                    "path_probability": None,
                    "tids": tids,
                    "path": " -> ".join(tids),
                }
            ]

    if len(old_top) < 3:
        # 重跑舊法枚舉僅取 top3
        gates.apply_llr_emb_gates(False)  # 舊門檻
        tracks_old = pes.load_tracks(str(merge_dir))
        all_paths, rej, n_e = pes.enumerate_paths(tracks_old)
        scored_old = []
        for path, edges in all_paths:
            scored_old.append(
                {
                    "tids": [t.tid for t in path],
                    "score": pes.path_score(path, edges),
                    "edges": edges,
                }
            )
        scored_old.sort(key=lambda p: -p["score"])
        # maximal
        tid_seqs = {tuple(p["tids"]) for p in scored_old}
        is_prefix = set()
        for q in tid_seqs:
            for k in range(1, len(q)):
                is_prefix.add(q[:k])
        maximal_old = [p for p in scored_old if tuple(p["tids"]) not in is_prefix]
        old_top = topk_metrics(maximal_old, gt_set, 3)

    # --- LLR-GT 修正前（無三修，但用 gt calib；節點關、超節點關、門檻舊）---
    print("=== LLR-GT 修正前 ===")
    gates.apply_llr_emb_gates(False)
    (
        _,
        scored_pre,
        maximal_pre,
        n_e_pre,
        _,
        srep_pre,
        gate_pre,
        opt_pre,
    ) = run_once(
        merge_dir,
        calib,
        use_emb_gate_fix=False,
        use_supernode=False,
        use_node_evidence=False,
    )
    pre_top = topk_metrics(maximal_pre, gt_set, 3)

    # --- 消融 ---
    ablations = {}
    configs = [
        ("fix1_only", dict(use_emb_gate_fix=True, use_supernode=False, use_node_evidence=False)),
        ("fix1_2", dict(use_emb_gate_fix=True, use_supernode=True, use_node_evidence=False)),
        ("fix1_2_3", dict(use_emb_gate_fix=True, use_supernode=True, use_node_evidence=True)),
    ]
    post_scored = None
    post_maximal = None
    post_srep = None
    post_gate = None
    for name, opts in configs:
        print(f"=== ablation {name} ===")
        tracks_r, scored, maximal, n_e, nodes, srep, gate, options = run_once(
            merge_dir, calib, **opts
        )
        top1 = maximal[0] if maximal else None
        pr1 = precision_recall(top1["tids"], gt_set) if top1 else None
        ablations[name] = {
            "options": options,
            "gate": gate,
            "n_paths_maximal": len(maximal),
            "n_legal_edges": n_e,
            "multi_only": srep.get("multi_only"),
            "top1": {
                "precision": pr1["precision"] if pr1 else None,
                "recall": pr1["recall"] if pr1 else None,
                "path_probability": top1.get("path_probability") if top1 else None,
                "score": top1["score"] if top1 else None,
                "tids": top1["tids"] if top1 else None,
                "super_labels": top1.get("super_labels") if top1 else None,
                "path": " -> ".join((top1.get("super_labels") or top1["tids"])) if top1 else None,
            },
            "top3": topk_metrics(maximal, gt_set, 3),
            "check_08_43": find_08_43_in_top10(maximal),
        }
        if name == "fix1_2_3":
            post_scored = scored
            post_maximal = maximal
            post_srep = srep
            post_gate = gate
            # 寫出正式輸出
            out_dir = out_root / "gt_calib_0507_fixed"
            out_dir.mkdir(parents=True, exist_ok=True)
            alt = llr.best_disjoint_alternative(maximal)
            tag = merge_dir.name
            llr.write_txt_report(
                out_dir / f"{tag}_llr_out.txt",
                merge_dir,
                tracks_r,
                scored,
                maximal,
                n_e,
                alt,
                super_report=srep,
                gate_info=gate,
            )
            summary = llr.build_summary_json(
                merge_dir,
                scored,
                maximal,
                None,
                alt,
                n_e,
                len(tracks_r),
                super_report=srep,
                gate_info=gate,
                options=options,
            )
            (out_dir / f"{tag}_llr_top1.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / f"{tag}_supernodes.json").write_text(
                json.dumps(srep, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    post_top = ablations["fix1_2_3"]["top3"]

    # GT best under post scoring — 亦精確計分預期鏈並併入比較
    expected_tids = EXPECTED_EXPAND
    # 在 scored 中找精確匹配或集合覆蓋
    gt_best = find_gt_best_path(post_scored, gt_set)
    # 若預期 11/11 鏈可行，構造其 score
    from path_enum_llr import (
        build_supernodes,
        _best_member_edge,
        path_score_llr,
        expand_path_tids,
        edge_llr,
    )

    # 精確重建預期超節點路徑分數
    gates.apply_llr_emb_gates(True)
    snodes, _ = build_supernodes(tracks)
    by_m = {}
    for sn in snodes:
        for tid in sn.tids:
            by_m[tid] = sn
    exp_groups = expected_groups
    chain_sns = [by_m[g[0]] for g in exp_groups]
    # 去重 sid order
    chain_unique = []
    seen_sid = set()
    for sn in chain_sns:
        if sn.sid not in seen_sid:
            chain_unique.append(sn)
            seen_sid.add(sn.sid)
    edges_e = []
    hist = [chain_unique[0].emb]
    chain_ok = True
    for i in range(len(chain_unique) - 1):
        sa, sb = chain_unique[i], chain_unique[i + 1]
        best, _ = _best_member_edge(sa, sb)
        if best is None:
            chain_ok = False
            break
        u, v, dt, hop, emb, h_dist = best
        class _P:
            pass
        p = _P()
        p.emb = sb.emb
        hsim = pes.hist_emb_sim(hist, p)
        need = pes.EMB_HIST_MIN - (
            0.02 if (h_dist is not None and h_dist < pes.H_DIST_GATE) else 0
        )
        if hsim < need:
            chain_ok = False
            break
        e = edge_llr(u, v, dt, emb, h_dist, calib)
        e["hop"] = hop
        e["hist_emb"] = hsim
        e["from_super"] = sa.label
        e["to_super"] = sb.label
        e["from_members"] = sa.tids
        e["to_members"] = sb.tids
        edges_e.append(e)
        hist.append(sb.emb)
    expected_path_rec = None
    if chain_ok:
        sc, nevs = path_score_llr(chain_unique, edges_e, calib)
        expected_path_rec = {
            "tids": expand_path_tids(
                # fake index expand
                chain_unique,
                list(range(len(chain_unique))),
            ),
            "super_labels": [s.label for s in chain_unique],
            "score": sc,
            "edges": edges_e,
            "node_evidence": nevs,
        }
        # expand_path_tids uses nodes[i] — chain_unique works as nodes list
        pr = precision_recall(expected_path_rec["tids"], gt_set)
        expected_path_rec.update(pr)

    if gt_best is None or (
        expected_path_rec
        and expected_path_rec.get("n_hit", 0) >= gt_best.get("n_hit", 0)
        and expected_path_rec.get("precision", 0) >= 1.0 - 1e-12
    ):
        if expected_path_rec and expected_path_rec.get("precision", 0) >= 1.0 - 1e-12:
            gt_best = {
                "tids": expected_path_rec["tids"],
                "score": expected_path_rec["score"],
                "precision": expected_path_rec["precision"],
                "recall": expected_path_rec["recall"],
                "n_hit": expected_path_rec["n_hit"],
                "n_path": expected_path_rec["n_path"],
                "super_labels": expected_path_rec["super_labels"],
                "source": "expected_super_chain",
            }

    gt_rank = None
    gt_best_P = None
    if gt_best:
        # rank among maximal by score
        better = sum(1 for p in post_maximal if p["score"] > gt_best["score"] + 1e-9)
        tied = [
            p
            for p in post_maximal
            if abs(p["score"] - gt_best["score"]) < 1e-9
            and p["tids"] == gt_best["tids"]
        ]
        gt_rank = better + 1
        # softmax among maximal + inject if missing
        scores = [p["score"] for p in post_maximal]
        if not any(p["tids"] == gt_best["tids"] for p in post_maximal):
            scores.append(gt_best["score"])
            from scipy.special import logsumexp
            import math as _math

            log_z = logsumexp(scores)
            gt_best_P = float(_math.exp(gt_best["score"] - log_z))
            gt_rank = better + 1  # among extended
            gt_best["softmax_note"] = "inject expected chain into beam Softmax"
        else:
            for p in post_maximal:
                if p["tids"] == gt_best["tids"]:
                    gt_best_P = p.get("path_probability")
                    break

    results = {
        "warning": "IN-SAMPLE：校準與評估同一資料集 0507，僅供診斷；正式效果需 0528 驗證。",
        "enumeration_note": (
            "修正一（EMB=0.80）後全圖合法邊≫80，極大路徑改 beam（width=64）近似 Softmax／Top-k；"
            "可行性與預期鏈為精確檢查。GT 子圖邊少時仍全枚舉。"
        ),
        "topology": topo,
        "supernode_feasibility": feas,
        "mislabel_08_43_super_check": mislabel_check,
        "all_multi_supernodes": srep.get("multi_only"),
        "all_supernodes": srep.get("supernodes"),
        "top3": {
            "old": old_top,
            "llr_gt_before_fix": pre_top,
            "llr_gt_after_fix": post_top,
        },
        "ablation": {
            k: {
                "top1_recall": v["top1"]["recall"],
                "top1_precision": v["top1"]["precision"],
                "top1_P": v["top1"]["path_probability"],
                "top1_path": v["top1"]["path"],
                "check_08_43": v["check_08_43"]["verdict"],
                "n_maximal": v["n_paths_maximal"],
                "n_edges": v["n_legal_edges"],
                "multi_only": v["multi_only"],
            }
            for k, v in ablations.items()
        },
        "ablation_detail": ablations,
        "gt_best_after_fix": {
            **(gt_best or {}),
            "rank_among_maximal": gt_rank,
            "path_probability": gt_best_P,
        },
        "sim_calib": {
            "sim_gt": calib.get("sim_gt"),
            "sim_nongt": calib.get("sim_nongt"),
        },
        "gate_after": post_gate,
    }

    out_json = out_root / "structural_fix_0507_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"JSON → {out_json}")

    # markdown report
    md = write_markdown(results)
    md_path = REPO_ROOT / "comparison_gt_20260507.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"MD → {md_path}")
    return results


def write_markdown(R: dict) -> str:
    lines = []
    lines.append("# GT 評估與結構修正報告：人員追蹤_20260507")
    lines.append("")
    lines.append(
        "> **本輪為 in-sample（校準與評估同一資料集 0507）。結論僅供診斷，正式效果需在 0528 上驗證。**"
    )
    lines.append(">")
    lines.append("> GT **未**進入硬規則或候選篩選；路徑枚舉不知道 GT 的存在。")
    lines.append(">")
    lines.append(
        "> **GT 更正：`K8-08_43` 為誤標，已剔除。GT=11。**  "
        "結構修正日期 **2026-07-15**（修正一／二／三）。"
    )
    lines.append("")
    lines.append("## 結構修正摘要（附依據）")
    lines.append("")
    lines.append("| 修正 | 內容 | 依據 |")
    lines.append("|------|------|------|")
    lines.append(
        "| 一 emb 門檻 | `EMB_EDGE/HIST_MIN`→**0.80**（runtime 覆寫，未改 `path_enum_scoring.py`） | "
        "emb\\|same≈N(0.917,0.023)；舊 0.91≈μ−0.3σ 構造性拒真轉移；0.80≈μ_diff−2.5σ |"
    )
    lines.append(
        "| 二 共存超節點 | 重疊≥0.5s 且（OVERLAP 或 H∧dH<80）；同鏡跳過；"
        "**單幀** track 落在對方區間內亦計共存 | "
        "GT 共存對應同一人；`22_19` 時長=0 需退化規則 |"
    )
    lines.append(
        "| 三 節點+dt | 節點 w·ln(P(sim\\|GT)/P(sim\\|nonGT))；transit σ→**1.0** PRIOR-WEAK | "
        "sim\\|GT n=11、sim\\|nonGT n=15；先驗放寬 |"
    )
    lines.append("")

    # topology
    lines.append("## 拓撲：上輪未答補完")
    lines.append("")
    for k, v in R["topology"].items():
        adj = "是（hop=1）" if v["adjacent_hop1"] else f"否（hop={v['hop_count']}）"
        lines.append(f"- **{k}**：{adj}；在 `ADJACENT`={'是' if v['in_ADJACENT'] else '否'}")
    lines.append("")

    # supernodes
    lines.append("## 修正二：實際形成的超節點")
    lines.append("")
    multi = R.get("all_multi_supernodes") or []
    if multi:
        for m in multi:
            lines.append(f"- `{{{', '.join(m)}}}`")
    else:
        lines.append("- （無多成員超節點）")
    lines.append("")
    mc = R["mislabel_08_43_super_check"]
    lines.append("### 錯標試金石：`K8-08_43`")
    if mc["merged_with_gt"]:
        lines.append(f"- **失敗**：併入含 GT 超節點：{mc['offending_supernodes']}")
    else:
        lines.append("- **通過**：未併入任何含 GT 的超節點（23↔08 無 OVERLAP、無 H）。")
    lines.append("")

    # feasibility
    F = R["supernode_feasibility"]
    lines.append("## 1. 超節點版 GT 可行性")
    lines.append("")
    lines.append("預期鏈：")
    lines.append("")
    lines.append("`09_7 → {08_30,01_7} → 07_40 → {23_8,22_19} → 22_22 → 07_112 → 01_50 → {08_77,01_62}`")
    lines.append("")
    lines.append(
        f"- 全鏈可行：**{'是 → 單路徑覆蓋上限 11/11' if F['chain_fully_feasible'] else '否'}**"
    )
    lines.append(f"- 單路徑覆蓋上限（沿預期鏈）：**{F['single_path_cover_upper']}/11**")
    if F.get("chain_fully_feasible"):
        lines.append(
            "- 計分排名上仍可能插入非 GT（如 `K8-09_42` 取代 `22_22`）；"
            "硬規則瓶頸已打開，剩餘是軟證據排序問題。"
        )
    lines.append("")
    lines.append("### 邊檢查")
    lines.append("")
    for e in F["edge_checks"]:
        if e.get("ok"):
            lines.append(
                f"- ✓ `{e['from']}`→`{e['to']}` via `{e['via']}`  "
                f"hop={e['hop']} dt={e['dt']:.1f}s emb={e['emb']:.3f}"
            )
        else:
            lines.append(f"- ✗ `{e.get('from')}`→`{e.get('to')}`")
            if e.get("example_edge_check"):
                ex = e["example_edge_check"]
                lines.append(
                    f"  - 例：`{ex['pair']}` reason=`{ex['reason']}` "
                    f"emb={ex.get('emb')} hop={ex.get('hop')} dt={ex.get('dt')}"
                )
            if e.get("reject_counts"):
                lines.append(f"  - 拒絕原因計數：`{e['reject_counts']}`")
    if F["hist_checks"].get("fails"):
        lines.append("")
        lines.append("歷史外觀失敗：")
        for h in F["hist_checks"]["fails"]:
            lines.append(f"- `{h['to']}` hist_emb={h['hist_emb']:.3f} < {h['need']}")
    lines.append("")

    # top3
    lines.append("## 2. 三套對照 Top-3（recall 分母=11）")
    lines.append("")
    for tag, key in [
        ("舊法", "old"),
        ("LLR-GT 修正前", "llr_gt_before_fix"),
        ("LLR-GT 修正後（一+二+三）", "llr_gt_after_fix"),
    ]:
        lines.append(f"### {tag}")
        lines.append("")
        lines.append("| # | prec | rec | P | 路徑 |")
        lines.append("|---|------|-----|---|------|")
        for r in R["top3"][key]:
            P = r.get("path_probability")
            Ps = f"{P:.4f}" if P is not None else "—"
            lines.append(
                f"| {r['rank']} | {r['precision']:.2f} | {r['recall']:.2f} | {Ps} | "
                f"`{r.get('path') or ' -> '.join(r.get('tids') or [])}` |"
            )
        lines.append("")

    # gt best
    lines.append("## 3. GT 最佳路徑（修正後）")
    lines.append("")
    gb = R["gt_best_after_fix"]
    if gb and gb.get("tids"):
        lines.append(f"- 路徑：`{' → '.join(gb['tids'])}`")
        lines.append(
            f"- precision={gb.get('precision'):.2f}  recall={gb.get('recall'):.3f}  "
            f"（hit={gb.get('n_hit')}/{N_GT}）"
        )
        lines.append(f"- 極大路徑中排名：**#{gb.get('rank_among_maximal')}**")
        if gb.get("path_probability") is not None:
            lines.append(f"- Softmax 機率：**{gb['path_probability']:.6f}**")
        else:
            lines.append("- Softmax 機率：該 tid 序列未進入極大路徑集合（見全枚舉 score 排名）。")
    else:
        lines.append("- （未找到 precision=1 的路徑）")
    lines.append("")

    # ablation
    lines.append("## 4. 消融（Top-1 recall）")
    lines.append("")
    lines.append("| 設定 | Top-1 recall | prec | P | 路徑 | 08_43∈Top10 |")
    lines.append("|------|-------------|------|---|------|-------------|")
    labels = {
        "fix1_only": "只開修正一",
        "fix1_2": "一+二",
        "fix1_2_3": "一+二+三",
    }
    for k, lab in labels.items():
        a = R["ablation"][k]
        P = a["top1_P"]
        Ps = f"{P:.4f}" if P is not None else "—"
        lines.append(
            f"| {lab} | **{a['top1_recall']:.2f}** | {a['top1_precision']:.2f} | {Ps} | "
            f"`{a['top1_path']}` | {a['check_08_43']} |"
        )
    lines.append("")

    # 08_43 detail
    lines.append("## 5. 修正一生效後：`K8-08_43` 是否出現在 Top-10")
    lines.append("")
    # prefer fix1_only check (修正一 only), also report full
    c1 = R["ablation_detail"]["fix1_only"]["check_08_43"]
    c123 = R["ablation_detail"]["fix1_2_3"]["check_08_43"]
    lines.append(f"- **只開修正一**：{c1['verdict']}")
    lines.append(f"- **一+二+三**：{c123['verdict']}")
    lines.append("")
    for label, c in [("只開修正一", c1), ("一+二+三", c123)]:
        if not c["present_in_top10"]:
            lines.append(f"### {label}：未出現")
            lines.append("")
            continue
        lines.append(f"### {label}：出現 — 證據分解")
        lines.append("")
        for h in c["hits"]:
            lines.append(
                f"**Rank #{h['rank']}** P={h.get('path_probability')}  "
                f"`{' → '.join(h.get('super_labels') or h['tids'])}`"
            )
            lines.append("")
            ne = h.get("node_evidence_08_43")
            if ne:
                lines.append(
                    f"- 節點 `08_43`：sim={ne.get('sim')} raw={ne.get('raw')} "
                    f"w={ne.get('w')} score={ne.get('score')}"
                )
            else:
                lines.append("- 節點證據：無獨立項（可能關節點或未命中）")
            lines.append("- 進入邊：")
            for e in h.get("in_edges") or []:
                lines.append(
                    f"  - `{e.get('from_super', e.get('from'))}`→`{e.get('to_super', e.get('to'))}` "
                    f"dt_model={e.get('dt_model')} LLR_dt={e.get('LLR_dt')} "
                    f"LLR_emb={e.get('LLR_emb')} LLR_dH={e.get('LLR_dH')} "
                    f"edge={e.get('score')} emb={e.get('emb')} dt={e.get('dt')}"
                )
            lines.append("- 離開邊：")
            for e in h.get("out_edges") or []:
                lines.append(
                    f"  - `{e.get('from_super', e.get('from'))}`→`{e.get('to_super', e.get('to'))}` "
                    f"dt_model={e.get('dt_model')} LLR_dt={e.get('LLR_dt')} "
                    f"LLR_emb={e.get('LLR_emb')} LLR_dH={e.get('LLR_dH')} "
                    f"edge={e.get('score')} emb={e.get('emb')} dt={e.get('dt')}"
                )
            lines.append("")
            lines.append("<details><summary>全路徑邊／節點</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(
                    {
                        "edges": h.get("all_edges"),
                        "node_evidence": h.get("all_node_evidence"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append("## 6. In-sample 警語")
    lines.append("")
    lines.append(R["warning"])
    lines.append("")
    if R.get("enumeration_note"):
        lines.append(f"**枚舉**：{R['enumeration_note']}")
        lines.append("")
    lines.append("## 產物")
    lines.append("")
    lines.append("| 檔案 | 說明 |")
    lines.append("|------|------|")
    lines.append("| `llr_gate_config.py` | 修正一門檻依據與覆寫 |")
    lines.append("| `path_enum_llr.py` | 超節點／節點證據／PRIOR-WEAK |")
    lines.append("| `../output/path_enum_llr/gt_calib_0507_fixed/` | 修正後正式輸出 |")
    lines.append("| `../output/path_enum_llr/structural_fix_0507_results.json` | 本輪數字全文 |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
