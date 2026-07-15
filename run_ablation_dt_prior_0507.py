# -*- coding: utf-8 -*-
"""
超節點 dt bug 修復對照 + dt-scoring / transition-prior 消融（0507）
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import llr_gate_config as gates  # noqa: E402
from evaluate_paths import load_gt, precision_recall, N_GT  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

TOP1_LABELS = [
    "K8-09_7",
    "{K8-08_30,K8-01_7}",
    "K8-07_40",
    "{K8-23_8,K8-22_19}",
    "K8-09_42",
    "K8-07_112",
    "K8-01_50",
    "{K8-08_77,K8-01_62}",
]
DIRECT_LABELS = [
    "K8-09_7",
    "{K8-08_30,K8-01_7}",
    "K8-07_40",
    "{K8-23_8,K8-22_19}",
    "K8-07_112",
    "K8-01_50",
    "{K8-08_77,K8-01_62}",
]
GT_CHAIN_LABELS = [
    "K8-09_7",
    "{K8-08_30,K8-01_7}",
    "K8-07_40",
    "{K8-23_8,K8-22_19}",
    "K8-22_22",
    "K8-07_112",
    "K8-01_50",
    "{K8-08_77,K8-01_62}",
]


def _old_member_edge(sa, sb):
    """修復前：edge_check 成員端點 dt，取 emb max。"""
    best = None
    for u in sa.members:
        for v in sb.members:
            if v.t_end < u.t_start - pes.DT_MAX:
                continue
            ok, reason, dt, hop, emb, h_dist = pes.edge_check(u, v)
            if ok and (best is None or emb > best[4]):
                best = (u, v, dt, hop, emb, h_dist)
    return best


def diagnose_dt_fix(tracks):
    supers, _ = llr.build_supernodes(tracks)
    by_m = {}
    for s in supers:
        for tid in s.tids:
            by_m[tid] = s

    # self-check
    sa = by_m["K8-23_8"]
    sb = by_m["K8-09_42"]
    sc = by_m["K8-07_112"]
    b1, _ = llr._best_member_edge(sa, sb)
    b2, _ = llr._best_member_edge(sb, sc)
    self_check = {
        "{23_8,22_19}->09_42": {
            "via": f"{b1[0].tid}->{b1[1].tid}" if b1 else None,
            "dt": float(b1[2]) if b1 else None,
            "expect": 20.400,
            "ok": b1 is not None and abs(b1[2] - 20.400) < 1e-3,
        },
        "09_42->07_112": {
            "via": f"{b2[0].tid}->{b2[1].tid}" if b2 else None,
            "dt": float(b2[2]) if b2 else None,
            "expect": 16.630,
            "ok": b2 is not None and abs(b2[2] - 16.630) < 1e-3,
        },
    }

    mismatched = []
    flipped = []
    n_old = n_new = 0
    old_keys = set()
    new_keys = set()
    for i, j in itertools.permutations(range(len(supers)), 2):
        sa, sb = supers[i], supers[j]
        old = _old_member_edge(sa, sb)
        new, _ = llr._best_member_edge(sa, sb)
        key = (sa.label, sb.label)
        if old:
            n_old += 1
            old_keys.add(key)
        if new:
            n_new += 1
            new_keys.add(key)
        if old and new and (
            len(sa.members) > 1 or len(sb.members) > 1
        ):
            if abs(old[2] - new[2]) > 1e-6:
                mismatched.append(
                    {
                        "from": sa.label,
                        "to": sb.label,
                        "via_old": f"{old[0].tid}->{old[1].tid}",
                        "via_new": f"{new[0].tid}->{new[1].tid}",
                        "dt_old": float(old[2]),
                        "dt_new": float(new[2]),
                        "emb_old": float(old[4]),
                        "emb_new": float(new[4]),
                    }
                )
        if bool(old) != bool(new):
            flipped.append(
                {
                    "from": sa.label,
                    "to": sb.label,
                    "old_ok": bool(old),
                    "new_ok": bool(new),
                    "dt_old": float(old[2]) if old else None,
                    "dt_new": float(new[2]) if new else None,
                    "via_old": f"{old[0].tid}->{old[1].tid}" if old else None,
                    "via_new": f"{new[0].tid}->{new[1].tid}" if new else None,
                }
            )

    return {
        "self_check": self_check,
        "n_legal_old": n_old,
        "n_legal_new": n_new,
        "n_dt_mismatch_multisuper": len(mismatched),
        "dt_mismatch_table": mismatched,
        "n_legality_flips": len(flipped),
        "legality_flips": flipped,
        "only_old": sorted(old_keys - new_keys),
        "only_new": sorted(new_keys - old_keys),
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
            }
        )
    return rows


def find_path_by_labels(maximal, labels):
    for i, p in enumerate(maximal, 1):
        if p.get("super_labels") == labels:
            return i, p
    # expand tid match
    expand = []
    for lab in labels:
        if lab.startswith("{"):
            expand.extend([x.strip() for x in lab[1:-1].split(",")])
        else:
            expand.append(lab)
    for i, p in enumerate(maximal, 1):
        if p["tids"] == expand:
            return i, p
    return None, None


def score_path_detail(maximal_entry):
    return {
        "score": maximal_entry["score"],
        "P": maximal_entry.get("path_probability"),
        "path": " -> ".join(maximal_entry.get("super_labels") or maximal_entry["tids"]),
    }


def main():
    merge_dir = (QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507").resolve()
    out_root = (OUTPUT_ROOT / "path_enum_llr").resolve()
    gt = load_gt(out_root / "ground_truth_20260507.json")
    gt_set = set(gt["person_tids"])

    # recalibrate with p_edge
    print("=== recalibrate ===")
    import calibrate_from_gt as cfg

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge_dir))
    gates.apply_llr_emb_gates(True)
    tracks = pes.load_tracks(str(merge_dir))
    samples = cfg.collect_gt_samples(
        tracks, gt["person_tids"], removed_mislabel=list(gt.get("removed_mislabel") or [])
    )
    calib = cfg.fit_calibration(samples)
    tp = cfg.compute_transition_prior(tracks, gt["person_tids"])
    calib["transition_prior"] = tp
    calib["dataset"] = merge_dir.name
    pkl = out_root / "calibration_gt0507.pkl"
    import pickle

    with pkl.open("wb") as f:
        pickle.dump(calib, f)
    hist = out_root / "emb_same_diff_hist_gt0507.png"
    cfg.save_emb_histogram(samples["emb_same"], samples["emb_diff"], hist)
    cfg.write_report(calib, samples, hist, out_root / "calibration_gt0507_report.txt")
    print(f"p_edge={tp['p_edge']} = {tp['n_gt_true_transitions']}/{tp['n_legal_edges']}")

    print("=== dt fix diagnose ===")
    gates.apply_llr_emb_gates(True)
    diag = diagnose_dt_fix(tracks)
    print("self_check", diag["self_check"])
    print(
        f"legal edges old={diag['n_legal_old']} new={diag['n_legal_new']}  "
        f"mismatch={diag['n_dt_mismatch_multisuper']}  flips={diag['n_legality_flips']}"
    )

    ablations = {}
    configs = [
        ("A", True, False),
        ("B", False, False),
        ("C", False, True),
        ("D", True, True),
    ]
    for name, dt_on, prior_on in configs:
        print(f"\n=== ablation {name}: dt={dt_on} prior={prior_on} ===")
        _, scored, maximal, n_e, nodes, srep, gate, opt = llr.run_llr(
            merge_dir,
            calib,
            use_emb_gate_fix=True,
            use_supernode=True,
            use_node_evidence=True,
            dt_scoring=dt_on,
            transition_prior=prior_on,
        )
        r_detour, p_detour = find_path_by_labels(maximal, TOP1_LABELS)
        r_direct, p_direct = find_path_by_labels(maximal, DIRECT_LABELS)
        r_gt, p_gt = find_path_by_labels(maximal, GT_CHAIN_LABELS)
        # if GT chain not in maximal (beam), reconstruct score among scored
        if p_gt is None:
            expand = []
            for lab in GT_CHAIN_LABELS:
                if lab.startswith("{"):
                    expand.extend([x.strip() for x in lab[1:-1].split(",")])
                else:
                    expand.append(lab)
            for i, p in enumerate(scored, 1):
                if p["tids"] == expand or p.get("super_labels") == GT_CHAIN_LABELS:
                    # rank among maximal by score
                    better = sum(1 for m in maximal if m["score"] > p["score"] + 1e-9)
                    r_gt = better + 1
                    p_gt = p
                    break

        gap = None
        if p_detour and p_direct:
            gap = float(p_detour["score"] - p_direct["score"])

        ablations[name] = {
            "options": opt,
            "n_legal_edges": n_e,
            "n_maximal": len(maximal),
            "top3": topk_rows(maximal, gt_set, 3),
            "detour_09_42": {
                "rank": r_detour,
                **(score_path_detail(p_detour) if p_detour else {}),
            },
            "direct": {
                "rank": r_direct,
                **(score_path_detail(p_direct) if p_direct else {}),
            },
            "score_gap_detour_minus_direct": gap,
            "gt_chain": {
                "rank": r_gt,
                **(score_path_detail(p_gt) if p_gt else {}),
            },
            "top1_path": " -> ".join(
                maximal[0].get("super_labels") or maximal[0]["tids"]
            )
            if maximal
            else None,
        }

    results = {
        "warning": "IN-SAMPLE 0507 only.",
        "transition_prior": tp,
        "dt_bugfix": diag,
        "ablation": ablations,
        "dt_scoring_rationale": gates.DT_SCORING_RATIONALE,
    }
    out_json = out_root / "ablation_dt_prior_0507.json"
    out_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"JSON → {out_json}")

    md = write_md(results)
    md_path = REPO_ROOT / "comparison_gt_20260507.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"MD → {md_path}")
    return results


def write_md(R: dict) -> str:
    D = R["dt_bugfix"]
    lines = []
    lines.append("# GT 評估與結構修正報告：人員追蹤_20260507")
    lines.append("")
    lines.append(
        "> **本輪為 in-sample（校準與評估同一資料集 0507）。結論僅供診斷，正式效果需在 0528 上驗證。**"
    )
    lines.append(">")
    lines.append("> GT **未**進入硬規則或候選篩選。")
    lines.append(">")
    lines.append(
        "> **2026-07-15**：超節點 dt 聯集修復；`--dt-scoring` / `--transition-prior` 消融。"
    )
    lines.append("")

    lines.append("## Bug 記錄：超節點 dt 用成員端點")
    lines.append("")
    lines.append(
        "修復前 `_best_member_edge` 呼叫 `edge_check(u,v)`，"
        "`dt = max(v.t_start − u.t_end, 0)` 取 **emb 最大成員對** 的端點，"
        "而非超節點聯集 `max(sb.t_start − sa.t_end, 0)`。"
    )
    lines.append("")
    lines.append("### 自查")
    lines.append("")
    for k, v in D["self_check"].items():
        tag = "✓" if v["ok"] else "✗"
        lines.append(
            f"- {tag} `{k}`：dt={v['dt']}（期待 {v['expect']}）via `{v['via']}`"
        )
    lines.append("")
    lines.append(
        f"- 合法邊數：修復前 **{D['n_legal_old']}** → 修復後 **{D['n_legal_new']}**"
    )
    lines.append(
        f"- 含超節點且 dt 不一致：**{D['n_dt_mismatch_multisuper']}** 條"
        f"（先前回報 13 條；本輪逐條如下）"
    )
    lines.append(
        f"- 合法性翻轉：**{D['n_legality_flips']}** 條"
        f"（僅舊有={len(D['only_old'])}，僅新有={len(D['only_new'])}）"
    )
    lines.append("")
    lines.append("### dt 不一致對照（修復前→後）")
    lines.append("")
    lines.append("| from → to | via_old | via_new | dt_old | dt_new |")
    lines.append("|-----------|---------|---------|--------|--------|")
    for r in D["dt_mismatch_table"]:
        lines.append(
            f"| `{r['from']}`→`{r['to']}` | `{r['via_old']}` | `{r['via_new']}` | "
            f"{r['dt_old']:.3f} | **{r['dt_new']:.3f}** |"
        )
    if not D["dt_mismatch_table"]:
        lines.append("| （無） | | | | |")
    lines.append("")
    if D["legality_flips"]:
        lines.append("### 合法性翻轉")
        lines.append("")
        for f in D["legality_flips"]:
            lines.append(
                f"- `{f['from']}`→`{f['to']}`：old_ok={f['old_ok']} → new_ok={f['new_ok']}  "
                f"dt {f['dt_old']}→{f['dt_new']}"
            )
        lines.append("")
    else:
        lines.append("### 合法性翻轉")
        lines.append("")
        lines.append("無。")
        lines.append("")

    tp = R["transition_prior"]
    lines.append("## 轉移先驗 p_edge")
    lines.append("")
    lines.append(
        f"`p_edge = {tp['n_gt_true_transitions']} / {tp['n_legal_edges']} = **{tp['p_edge']:.6f}**`  "
        f"（`ln(p)={tp.get('ln_p_edge'):.4f}`）"
    )
    lines.append("")
    lines.append(R["dt_scoring_rationale"])
    lines.append("")

    lines.append("## 消融矩陣（修復一之後）")
    lines.append("")
    lines.append("| 組 | dt-scoring | transition-prior |")
    lines.append("|----|------------|------------------|")
    lines.append("| A | on | off |")
    lines.append("| B | off | off |")
    lines.append("| C | off | on |")
    lines.append("| D | on | on |")
    lines.append("")

    for name in ("A", "B", "C", "D"):
        a = R["ablation"][name]
        lines.append(f"### 組 {name}（dt={a['options']['dt_scoring']}, prior={a['options']['transition_prior']}）")
        lines.append("")
        lines.append("| # | prec | rec | P | 路徑 |")
        lines.append("|---|------|-----|---|------|")
        for r in a["top3"]:
            P = r["path_probability"]
            Ps = f"{P:.4f}" if P is not None else "—"
            lines.append(
                f"| {r['rank']} | {r['precision']:.2f} | {r['recall']:.2f} | {Ps} | `{r['path']}` |"
            )
        lines.append("")
        det = a["detour_09_42"]
        di = a["direct"]
        gt = a["gt_chain"]
        gap = a["score_gap_detour_minus_direct"]
        lines.append(
            f"- 繞經 `09_42`：rank=#{det.get('rank')} score={det.get('score')} P={det.get('P')}"
        )
        lines.append(
            f"- 直連：rank=#{di.get('rank')} score={di.get('score')} P={di.get('P')}"
        )
        lines.append(f"- 分差（繞−直）：**{gap}**")
        lines.append(
            f"- GT 全鏈：rank=#{gt.get('rank')} score={gt.get('score')} P={gt.get('P')}"
        )
        lines.append("")

    A = R["ablation"]["A"]
    B = R["ablation"]["B"]
    C = R["ablation"]["C"]
    lines.append("## 重點比較")
    lines.append("")
    lines.append("### A vs B（移除 dt 軟計分）")
    lines.append("")
    a1 = A["top3"][0]
    b1 = B["top3"][0]
    same = a1["path"] == b1["path"]
    lines.append(f"- Top-1 路徑維持：**{'是' if same else '否'}**")
    lines.append(f"  - A：`{a1['path']}`  P={a1['path_probability']:.4f} rec={a1['recall']:.2f}")
    lines.append(f"  - B：`{b1['path']}`  P={b1['path_probability']:.4f} rec={b1['recall']:.2f}")
    if a1["path_probability"] is not None and b1["path_probability"] is not None:
        lines.append(
            f"- Top-1 機率變化：{a1['path_probability']:.4f} → {b1['path_probability']:.4f}  "
            f"（Δ={b1['path_probability']-a1['path_probability']:+.4f}）"
        )
    lines.append(
        f"- 繞−直 分差：A={A['score_gap_detour_minus_direct']} → B={B['score_gap_detour_minus_direct']}"
    )
    lines.append("")
    lines.append("### C（dt off + prior on）：是否解決繞路")
    lines.append("")
    c1 = C["top3"][0]
    lines.append(f"- Top-1：`{c1['path']}`  rec={c1['recall']:.2f} prec={c1['precision']:.2f}")
    has_942 = "09_42" in (c1["path"] or "")
    lines.append(f"- Top-1 是否仍含 `09_42`：**{'是' if has_942 else '否'}**")
    lines.append(
        f"- 繞經 rank=#{C['detour_09_42'].get('rank')} / 直連 rank=#{C['direct'].get('rank')}  "
        f"分差={C['score_gap_detour_minus_direct']}"
    )
    lines.append(
        f"- GT 全鏈 rank=#{C['gt_chain'].get('rank')}；Top-1 recall={c1['recall']:.2f}"
        f"（相對 A rec={a1['recall']:.2f}）"
    )
    lines.append("")
    lines.append("## In-sample 警語")
    lines.append("")
    lines.append(R["warning"])
    lines.append("")
    lines.append("## 產物")
    lines.append("")
    lines.append("| 檔案 | 說明 |")
    lines.append("|------|------|")
    lines.append("| `path_enum_llr.py` | 聯集 dt；`--dt-scoring`；`--transition-prior` |")
    lines.append("| `llr_gate_config.py` | emb 門檻 + dt 停用依據 |")
    lines.append("| `calibrate_from_gt.py` | `p_edge` 寫入 pkl |")
    lines.append("| `../output/path_enum_llr/ablation_dt_prior_0507.json` | 本輪全文 |")
    lines.append("| `../output/path_enum_llr/calibration_gt0507.pkl` | 含 transition_prior |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
