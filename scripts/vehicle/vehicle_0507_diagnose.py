#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
車輛 0507 診斷（純輸出，不調參）
1) 身分分 sim 排序健檢
2) GT 路徑合法性 + 假設池存活
3) Top-1 逐邊帳目
4) beam 剪枝判定
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKING = _REPO_ROOT / "scripts" / "tracking"
for _p in (_REPO_ROOT, _TRACKING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import track_path_m0 as tp

OUT = tp.OUTPUT_ROOT / "v1.0"
MERGE = tp.QUERY_FILTER_OUTPUT_ROOT / "車輛追蹤_20260507"
GT_PATH = OUT / "ground_truth_vehicle_20260507.json"
CALIB = OUT / "calibration_vehicle_gt0507.pkl"
RUN_DIR = OUT / "vehicle_m0_0507"
DIAG_DIR = OUT / "vehicle_0507_diagnose"
SIM_MIN = 0.85


def _pick_crop(merge: Path, tid: str) -> Path | None:
    cam, tid_s = tid.rsplit("_", 1)
    try:
        _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
    except Exception:
        return None
    if not crops:
        return None
    return tp._pick_rep_crop(crops)


def section_identity(tracks, gt_set: set[str], lines: list[str]) -> list[dict]:
    rows = sorted(tracks, key=lambda t: (-float(t.sim), t.tid))
    lines.append("## 1. 身分分健檢（66 候選按 sim 降序）")
    lines.append("")
    lines.append("| rank | tid | cam | sim | tag | t_start | t_end |")
    lines.append("|---:|---|---|---:|---|---:|---:|")
    gt_ranks = []
    front_nongt = []  # non-GT ranking ahead of last GT in top band
    for i, t in enumerate(rows, 1):
        tag = "GT" if t.tid in gt_set else "nonGT"
        lines.append(
            f"| {i} | `{t.tid}` | {t.cam} | {float(t.sim):.4f} | **{tag}** | "
            f"{t.t_start:.1f} | {t.t_end:.1f} |"
        )
        if t.tid in gt_set:
            gt_ranks.append((i, t.tid, float(t.sim)))
        else:
            front_nongt.append((i, t))

    lines.append("")
    lines.append("### GT 12 條排名落點")
    lines.append("")
    lines.append("| GT tid | rank | sim |")
    lines.append("|---|---:|---:|")
    for r, tid, sim in gt_ranks:
        lines.append(f"| `{tid}` | {r} | {sim:.4f} |")
    ranks_only = [r for r, _, _ in gt_ranks]
    lines.append("")
    lines.append(
        f"- GT rank 範圍：min={min(ranks_only)}  max={max(ranks_only)}  "
        f"median≈{sorted(ranks_only)[len(ranks_only)//2]}"
    )
    # 「前段」：取排在最差 GT 之前的所有非 GT，或更實用：前 20 / 高於最差 GT
    worst_gt_rank = max(ranks_only)
    best_gt_rank = min(ranks_only)
    ahead = [(r, t) for r, t in front_nongt if r < worst_gt_rank]
    lines.append(
        f"- 最優 GT 排名=#{best_gt_rank}；最差 GT 排名=#{worst_gt_rank}；"
        f"其間／之前的非 GT 共 {len(ahead)} 條"
    )
    # 特別標「擠進前段」：排名優於中位 GT、或前 15
    mid_gt = sorted(ranks_only)[len(ranks_only) // 2]
    intruders = [(r, t) for r, t in front_nongt if r <= mid_gt]
    lines.append(f"- 非 GT 排名 ≤ GT 中位 rank({mid_gt}) 者（前段侵入）：**{len(intruders)}** 條")
    lines.append("")

    crop_dir = DIAG_DIR / "front_nongt_crops"
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True)
    lines.append("### 前段非 GT 代表 crop")
    lines.append("")
    lines.append("| rank | tid | sim | rep_crop |")
    lines.append("|---:|---|---:|---|")
    for r, t in intruders:
        rep = _pick_crop(MERGE, t.tid)
        dest = None
        if rep and rep.is_file():
            dest = crop_dir / f"rank{r:02d}_{t.tid}_{rep.name}"
            shutil.copy2(rep, dest)
        lines.append(
            f"| {r} | `{t.tid}` | {float(t.sim):.4f} | "
            f"{('`'+str(dest)+'`') if dest else '（無 crop）'} |"
        )
    return [{"rank": r, "tid": tid, "sim": sim} for r, tid, sim in gt_ranks]


def gt_chronological_edges(gt_tracks: list) -> list[dict]:
    ordered = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))
    rows = []
    for u, v in zip(ordered, ordered[1:]):
        dt_raw = v.t_start - u.t_end
        dt = max(dt_raw, 0.0)
        ok, reason, dt_e, hop, emb, h_dist = tp.edge_check(u, v)
        flag = ""
        if dt > tp.DT_MAX:
            flag = "★超 DT_MAX"
        rows.append(
            {
                "from": u.tid,
                "to": v.tid,
                "ok": bool(ok),
                "reason": reason or "ok",
                "dt_raw": float(dt_raw),
                "dt": float(dt),
                "hop": hop,
                "emb": float(emb) if emb else 0.0,
                "h_dist": float(h_dist) if h_dist is not None else None,
                "flag": flag,
                "u": u,
                "v": v,
            }
        )
    return ordered, rows


def longest_legal_subchains(ordered: list, edge_rows: list) -> list[list[str]]:
    """連續 edge_check 合法的最長子鏈（可多條等長）。"""
    n = len(ordered)
    ok = [e["ok"] for e in edge_rows]
    chains = []
    i = 0
    while i < n:
        j = i
        while j < n - 1 and ok[j]:
            j += 1
        chains.append([ordered[k].tid for k in range(i, j + 1)])
        i = j + 1
    max_len = max(len(c) for c in chains) if chains else 0
    return [c for c in chains if len(c) == max_len]


def map_tid_to_super(nodes, tid: str):
    for i, sn in enumerate(nodes):
        if tid in sn.tids:
            return i, sn
    return None, None


def score_super_path(nodes, path_idx, calib, dt_scoring=False, transition_prior=False):
    """對一條超節點 index 路徑重算邊＋節點分（需 succ 合法）。"""
    succ, _, _ = tp._build_succ(nodes)
    succ_map = defaultdict(dict)
    for i, outs in enumerate(succ):
        for j, u, v, dt, hop, emb, h_dist in outs:
            succ_map[i][j] = (u, v, dt, hop, emb, h_dist)

    edges_info = []
    hist_embs = [nodes[path_idx[0]].emb]
    for a, b in zip(path_idx, path_idx[1:]):
        if b not in succ_map[a]:
            return None, None, f"無合法超節點邊 {nodes[a].label} -> {nodes[b].label}"
        u, v, dt, hop, emb, h_dist = succ_map[a][b]
        ok_h, hsim, emb_need = tp._hist_ok(hist_embs, nodes, b, h_dist)
        if not ok_h:
            return (
                None,
                None,
                f"歷史不像 {nodes[a].label}->{nodes[b].label} hist={hsim:.3f}<{emb_need}",
            )
        e = tp._make_edge_rec(
            nodes,
            a,
            b,
            u,
            v,
            dt,
            hop,
            emb,
            h_dist,
            hsim,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
        )
        edges_info.append(e)
        hist_embs.append(nodes[b].emb)
    sn_path = [nodes[i] for i in path_idx]
    score, node_evs = tp.path_score_llr(sn_path, edges_info, calib)
    return score, {"edges": edges_info, "node_evidence": node_evs, "tids": tp.expand_path_tids(nodes, path_idx), "super_labels": [nodes[i].label for i in path_idx]}, None


def beam_trace_gt_prefix(
    nodes,
    succ,
    calib,
    gt_path_idx: list[int],
    *,
    beam_width: int = 64,
    dt_scoring: bool = False,
    transition_prior: bool = False,
) -> dict:
    """
    模擬 beam：每層看 GT 前綴是否仍在 top-beam_width。
    回傳各深度存活狀態。
    """
    n = len(nodes)

    def _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim):
        return tp._make_edge_rec(
            nodes, idx, j, u, v, dt, hop, emb, h_dist, hsim, calib,
            dt_scoring=dt_scoring, transition_prior=transition_prior,
        )

    # 也算 GT 各前綴的邊分（不含節點分；beam 內部用邊分累加）
    # 注意：實際 beam 用的是 edge score only（見 enumerate_paths_llr）
    depth_reports = []
    # Start from GT's first node only? Beam starts from EVERY s.
    # GT path must appear as extension from its start node s=gt_path_idx[0].
    s0 = gt_path_idx[0]
    beam = [(0.0, [s0], [], [nodes[s0].emb])]
    alive = True
    for depth in range(1, len(gt_path_idx)):
        target = tuple(gt_path_idx[: depth + 1])
        nxt = []
        for sc, path_idx, edges_info, hist_embs in beam:
            idx = path_idx[-1]
            for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                if j in path_idx:
                    continue
                ok_h, hsim, _ = tp._hist_ok(hist_embs, nodes, j, h_dist)
                if not ok_h:
                    continue
                e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                nxt.append(
                    (
                        sc + e["score"],
                        path_idx + [j],
                        edges_info + [e],
                        hist_embs + [nodes[j].emb],
                    )
                )
        if not nxt:
            depth_reports.append(
                {
                    "depth": depth,
                    "target": [nodes[i].label for i in target],
                    "alive": False,
                    "reason": "該層無任何擴展（非 beam 剪枝）",
                    "gt_rank_in_nxt": None,
                    "n_nxt": 0,
                    "gt_edge_score": None,
                    "cutoff_score": None,
                }
            )
            alive = False
            break
        nxt.sort(key=lambda x: -x[0])
        # find GT prefix rank among ALL nxt before truncate
        gt_rank = None
        gt_sc = None
        for r, item in enumerate(nxt, 1):
            if tuple(item[1]) == target:
                gt_rank = r
                gt_sc = item[0]
                break
        kept = nxt[:beam_width]
        cutoff = kept[-1][0] if kept else None
        in_beam = any(tuple(x[1]) == target for x in kept)
        depth_reports.append(
            {
                "depth": depth,
                "target": [nodes[i].label for i in target],
                "alive": bool(in_beam),
                "reason": (
                    "存活"
                    if in_beam
                    else (
                        "被 beam 剪掉"
                        if gt_rank is not None
                        else "該層候選中無此 GT 前綴（邊／hist 不通或未從正確起點展開）"
                    )
                ),
                "gt_rank_in_nxt": gt_rank,
                "n_nxt": len(nxt),
                "gt_edge_score": gt_sc,
                "cutoff_score": cutoff,
                "beam_width": beam_width,
            }
        )
        if not in_beam:
            alive = False
            break
        beam = kept
    return {"alive_to_end": alive, "depths": depth_reports, "start": nodes[s0].label}


def section_gt_survival(tracks, gt_tids, calib, lines: list[str]):
    by_tid = {t.tid: t for t in tracks}
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    ordered, edge_rows = gt_chronological_edges(gt_tracks)

    lines.append("## 2. 正解存活檢查")
    lines.append("")
    lines.append("### 2.1 手工 GT 時間序路徑（track 級 edge_check）")
    lines.append("")
    lines.append("時間序：")
    lines.append("")
    lines.append("`" + " -> ".join(t.tid for t in ordered) + "`")
    lines.append("")
    lines.append("| # | from → to | dt | hop | emb | ok | reason | flag |")
    lines.append("|---:|---|---:|---:|---:|---|---|---|")
    for i, e in enumerate(edge_rows, 1):
        hop_s = e["hop"] if e["hop"] is not None else "—"
        lines.append(
            f"| {i} | `{e['from']}` → `{e['to']}` | {e['dt']:.1f} | {hop_s} | "
            f"{e['emb']:.3f} | {'✓' if e['ok'] else '✗'} | {e['reason']} | {e['flag'] or '—'} |"
        )

    chains = longest_legal_subchains(ordered, edge_rows)
    max_len = len(chains[0]) if chains else 0
    lines.append("")
    lines.append(f"### 2.2 最大合法子鏈（連續 edge_ok；長度={max_len}）")
    lines.append("")
    for c in chains:
        lines.append(f"- `{' -> '.join(c)}`")

    # 超節點圖：GT 映射
    tp.apply_llr_emb_gates(True)
    nodes, srep = tp._build_nodes(tracks, use_supernode=True)
    succ, rejected, n_legal = tp._build_succ(nodes)

    tid2si = {}
    for i, sn in enumerate(nodes):
        for tid in sn.tids:
            tid2si[tid] = i

    lines.append("")
    lines.append("### 2.3 GT → 超節點映射")
    lines.append("")
    lines.append("| GT tid | super | members |")
    lines.append("|---|---|---|")
    gt_super_seq = []
    seen_si = []
    for t in ordered:
        si = tid2si.get(t.tid)
        lab = nodes[si].label if si is not None else "MISSING"
        mem = nodes[si].tids if si is not None else []
        lines.append(f"| `{t.tid}` | `{lab}` | {mem} |")
        if si is not None and (not seen_si or seen_si[-1] != si):
            seen_si.append(si)
            gt_super_seq.append(si)

    lines.append("")
    lines.append(
        "去重後超節點時間序：`"
        + " -> ".join(nodes[i].label for i in gt_super_seq)
        + "`"
    )

    # 超節點相鄰邊檢查
    lines.append("")
    lines.append("### 2.4 超節點相鄰邊（聯集 dt / _best_member_edge）")
    lines.append("")
    lines.append("| from_super → to_super | ok | via | dt | hop | emb |")
    lines.append("|---|---|---|---:|---:|---:|")
    super_edge_ok = []
    for a, b in zip(gt_super_seq, gt_super_seq[1:]):
        best, _ = tp._best_member_edge(nodes[a], nodes[b])
        if best is None:
            lines.append(
                f"| `{nodes[a].label}` → `{nodes[b].label}` | ✗ | — | — | — | — |"
            )
            super_edge_ok.append(False)
        else:
            u, v, dt, hop, emb, h_dist = best
            lines.append(
                f"| `{nodes[a].label}` → `{nodes[b].label}` | ✓ | "
                f"`{u.tid}->{v.tid}` | {dt:.1f} | {hop} | {emb:.3f} |"
            )
            super_edge_ok.append(True)

    # 最大合法超節點子鏈
    s_chains = []
    i = 0
    while i < len(gt_super_seq):
        j = i
        while j < len(gt_super_seq) - 1 and super_edge_ok[j]:
            j += 1
        s_chains.append(gt_super_seq[i : j + 1])
        i = j + 1
    max_s = max(len(c) for c in s_chains) if s_chains else 0
    best_super_chains = [c for c in s_chains if len(c) == max_s]
    lines.append("")
    lines.append(f"最大合法超節點子鏈長度={max_s}：")
    for c in best_super_chains:
        lines.append(f"- `{' -> '.join(nodes[i].label for i in c)}`")

    # 跑 beam 枚舉（與 M0 相同設定）並搜尋
    lines.append("")
    lines.append("### 2.5 假設池搜尋（M0 凍結：beam / dt_scoring=off）")
    lines.append("")

    all_paths, _, n_legal_edges, nodes2, super_report = tp.enumerate_paths_llr(
        tracks,
        calib,
        use_supernode=True,
        dt_scoring=False,
        transition_prior=False,
        force_full=False,
    )
    nodes = nodes2
    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        score, node_evs = tp.path_score_llr(sn_path, edges_info, calib)
        scored.append(
            {
                "path_idx": path_idx,
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": node_evs,
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    single_maximal = tp.maximal_paths(scored)
    ranked, rank_meta = tp.build_ranked_hypotheses(
        single_maximal, nodes, tracks, calib, dt_scoring=False, transition_prior=False
    )
    top1 = ranked[0] if ranked else None
    top1_score = float(top1["score"]) if top1 else None

    def find_in_pool(target_supers: list[str], pool):
        """精確匹配 super_labels；另回報包含為子序列的最佳。"""
        exact = None
        best_sub = None
        tgt = tuple(target_supers)
        tgt_set_seq = target_supers
        for rank, p in enumerate(pool, 1):
            labs = p.get("super_labels") or []
            if tuple(labs) == tgt:
                exact = (rank, p)
                break
        for rank, p in enumerate(pool, 1):
            labs = p.get("super_labels") or []
            # subsequence match
            k = 0
            for lab in labs:
                if k < len(tgt_set_seq) and lab == tgt_set_seq[k]:
                    k += 1
            if k == len(tgt_set_seq):
                if best_sub is None or p["score"] > best_sub[1]["score"]:
                    best_sub = (rank, p)
        return exact, best_sub

    # 搜尋完整 GT 超節點序（多半不合法）
    full_labs = [nodes[i].label for i in gt_super_seq]
    exact_full, sub_full = find_in_pool(full_labs, ranked)

    lines.append(f"- beam 葉路徑數（scored）={len(scored)}；排名假設={len(ranked)}")
    lines.append(f"- Top-1 score={top1_score:.4f}  path=`{top1.get('path') if top1 else None}`")
    lines.append("")
    lines.append("**完整 GT 超節點序**（即使中間有斷邊，仍查池）：")
    if exact_full:
        r, p = exact_full
        lines.append(
            f"- 精確命中排名假設 #{r}  score={p['score']:.4f}  "
            f"Δ(Top1−GT)={top1_score - p['score']:+.4f}"
        )
    else:
        lines.append("- 精確命中：無")
    if sub_full:
        r, p = sub_full
        lines.append(
            f"- 作為子序列出現於排名假設 #{r}  score={p['score']:.4f}  "
            f"`{' -> '.join(p.get('super_labels') or [])}`"
        )
    else:
        lines.append("- 子序列命中：無")

    # 對每條最大合法超節點子鏈計分＋查池
    lines.append("")
    lines.append("**最大合法超節點子鏈 vs 假設池：**")
    lines.append("")
    chain_results = []
    for c in best_super_chains:
        labs = [nodes[i].label for i in c]
        sc, detail, err = score_super_path(
            nodes, c, calib, dt_scoring=False, transition_prior=False
        )
        exact, sub = find_in_pool(labs, ranked)
        # also search single_maximal / scored leaves
        exact_leaf, sub_leaf = find_in_pool(labs, scored)
        entry = {
            "labs": labs,
            "score": sc,
            "err": err,
            "exact_rank": exact[0] if exact else None,
            "exact_leaf_rank": exact_leaf[0] if exact_leaf else None,
            "detail": detail,
        }
        chain_results.append(entry)
        lines.append(f"- 子鏈 `{' -> '.join(labs)}`")
        if err:
            lines.append(f"  - 計分失敗：{err}")
        else:
            lines.append(f"  - 手工計分（node+edge）= **{sc:.4f}**")
            if top1_score is not None:
                lines.append(f"  - vs Top-1：Δ(Top1−GT)={top1_score - sc:+.4f}")
        if exact:
            lines.append(f"  - 排名假設池精確命中：**#{exact[0]}** score={exact[1]['score']:.4f}")
        else:
            lines.append("  - 排名假設池精確命中：無")
        if exact_leaf:
            lines.append(
                f"  - beam 葉／scored 精確命中：#{exact_leaf[0]} "
                f"score={exact_leaf[1]['score']:.4f}"
            )
        else:
            lines.append("  - beam 葉／scored 精確命中：無")
        if sub and not exact:
            lines.append(
                f"  - 子序列出現於排名 #{sub[0]}：`{' -> '.join(sub[1].get('super_labels') or [])}`"
            )

    # 另：在 scored 中找 GT recall 最高路徑
    def gt_cover(tids):
        return sum(1 for t in tids if t in set(gt_tids))

    best_rec = max(scored, key=lambda p: (gt_cover(p["tids"]), p["score"])) if scored else None
    if best_rec:
        # rank in ranked if present
        br = None
        for i, p in enumerate(ranked, 1):
            if tuple(p.get("super_labels") or []) == tuple(best_rec["super_labels"]):
                br = i
                break
        lines.append("")
        lines.append(
            f"**beam 池內 GT 覆蓋最高路徑**：cover={gt_cover(best_rec['tids'])}/12  "
            f"score={best_rec['score']:.4f}  "
            f"排名假設位次={br or '（未進排名／被矛盾作廢）'}  "
            f"`{' -> '.join(best_rec['super_labels'])}`"
        )

    return {
        "ordered": ordered,
        "edge_rows": edge_rows,
        "chains": chains,
        "gt_super_seq": gt_super_seq,
        "best_super_chains": best_super_chains,
        "nodes": nodes,
        "succ": succ,
        "scored": scored,
        "ranked": ranked,
        "top1": top1,
        "top1_score": top1_score,
        "chain_results": chain_results,
        "super_report": super_report,
        "n_legal": n_legal_edges,
    }


def section_top1_edges(ctx, gt_set, lines: list[str]):
    lines.append("## 3. Top-1 逐邊帳目（它靠什麼贏）")
    lines.append("")
    top1 = ctx["top1"]
    if not top1:
        lines.append("（無 Top-1）")
        return
    # ranked top1 may lack full edges — reload from scored match
    edges = top1.get("edges")
    node_ev = top1.get("node_evidence")
    if not edges:
        labs = tuple(top1.get("super_labels") or [])
        for p in ctx["scored"]:
            if tuple(p["super_labels"]) == labs:
                edges = p["edges"]
                node_ev = p["node_evidence"]
                break
    # fallback parse out.txt
    if not edges:
        from vehicle_0507_calibrate_run import parse_top1_from_out_txt

        node_ev, edges = parse_top1_from_out_txt(RUN_DIR / "0507_out.txt")

    lines.append(f"path: `{' -> '.join(top1.get('super_labels') or top1.get('tids') or [])}`")
    lines.append(f"score={top1['score']:.4f}  P={top1.get('path_probability')}")
    lines.append("")
    lines.append("### 節點身分分（node evidence）")
    lines.append("")
    lines.append("| super | sim | LLR_raw | w | node_score | GT? |")
    lines.append("|---|---:|---:|---:|---:|---|")
    sum_node = 0.0
    for ne in node_ev or []:
        sid = ne.get("super")
        sc = float(ne.get("score") or 0)
        sum_node += sc
        tag = "GT" if sid in gt_set or any(
            m in gt_set for m in (ne.get("members") or [])
        ) else "nonGT"
        # also check members via label tid
        if sid in gt_set:
            tag = "GT"
        lines.append(
            f"| `{sid}` | {float(ne.get('sim') or 0):.3f} | "
            f"{float(ne.get('raw') or 0):+.3f} | {float(ne.get('w') or 0):.3f} | "
            f"{sc:+.3f} | {tag} |"
        )
    lines.append(f"| Σ | | | | **{sum_node:+.3f}** | |")
    lines.append("")
    lines.append("### 邊（emb / LLR / 幾何）")
    lines.append("")
    lines.append(
        "| from → to | hop | dt | emb | LLR_emb | LLR_dH | LLR_dt | edge | d_H |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---|")
    sum_edge = 0.0
    for e in edges or []:
        fr = e.get("from_super") or e.get("from")
        to = e.get("to_super") or e.get("to")
        sc = float(e.get("score") or 0)
        sum_edge += sc
        ldh = e.get("LLR_dH")
        ldh_s = f"{ldh:+.3f}" if ldh is not None else "—"
        ldt = e.get("LLR_dt")
        if ldt is None or ldt in ("removed", "n/a", "—", "off"):
            ldt_s = "removed/off"
        else:
            try:
                ldt_s = f"{float(ldt):+.3f}"
            except (TypeError, ValueError):
                ldt_s = str(ldt)
        hd = e.get("h_dist")
        hd_s = f"{hd:.1f}" if hd is not None else "—"
        lines.append(
            f"| `{fr}` → `{to}` | {e.get('hop')} | {float(e.get('dt') or 0):.1f} | "
            f"{float(e.get('emb') or 0):.3f} | {float(e.get('LLR_emb') or 0):+.3f} | "
            f"{ldh_s} | {ldt_s} | {sc:+.3f} | {hd_s} |"
        )
    lines.append(f"| Σ | | | | | | | **{sum_edge:+.3f}** | |")
    lines.append("")
    lines.append(
        f"**合計驗算**：node Σ {sum_node:+.3f} + edge Σ {sum_edge:+.3f} = "
        f"**{sum_node + sum_edge:+.3f}**（Top-1 score={float(top1['score']):+.4f}）"
    )
    lines.append("")
    # 贏面摘要
    pos_edges = sorted(
        [e for e in (edges or []) if float(e.get("score") or 0) > 0.5],
        key=lambda e: -float(e.get("score") or 0),
    )
    lines.append("### 贏面摘要")
    lines.append("")
    if pos_edges:
        lines.append("高分邊（edge>0.5）：")
        for e in pos_edges[:5]:
            fr = e.get("from_super") or e.get("from")
            to = e.get("to_super") or e.get("to")
            lines.append(
                f"- `{fr}→{to}` emb={float(e.get('emb') or 0):.3f} "
                f"LLR_emb={float(e.get('LLR_emb') or 0):+.3f} "
                f"edge={float(e.get('score') or 0):+.3f}"
            )
    pos_nodes = sorted(
        [ne for ne in (node_ev or []) if float(ne.get("score") or 0) > 0.5],
        key=lambda ne: -float(ne.get("score") or 0),
    )
    if pos_nodes:
        lines.append("高身分分節點（node>0.5）：")
        for ne in pos_nodes:
            lines.append(
                f"- `{ne.get('super')}` sim={float(ne.get('sim') or 0):.3f} "
                f"node={float(ne.get('score') or 0):+.3f}"
            )


def section_beam(ctx, calib, lines: list[str]):
    lines.append("## 4. beam 健檢（width=64）")
    lines.append("")
    enum = (ctx["super_report"].get("enumeration") or {})
    lines.append(
        f"- 合法邊={ctx['n_legal']}  mode={enum.get('mode')}  "
        f"beam_width={enum.get('beam_width')}  n_beam_leaves={enum.get('n_beam_leaves')}"
    )
    lines.append("")

    nodes = ctx["nodes"]
    succ = ctx["succ"]
    # rebuild succ on same nodes
    succ, _, _ = tp._build_succ(nodes)

    for c in ctx["best_super_chains"]:
        labs = [nodes[i].label for i in c]
        lines.append(f"### 目標合法子鏈 `{' -> '.join(labs)}`")
        lines.append("")
        # 先確認整條在 succ 上可走
        ok_path = True
        for a, b in zip(c, c[1:]):
            found = any(j == b for j, *_ in succ[a])
            if not found:
                ok_path = False
                lines.append(
                    f"- 硬結論：超節點邊 `{nodes[a].label}→{nodes[b].label}` 不在 succ "
                    f"→ **無合法路徑**（非 beam 剪枝）"
                )
                break
        if not ok_path:
            continue

        # 是否在 scored 葉中
        in_scored = any(
            tuple(p["super_labels"]) == tuple(labs) for p in ctx["scored"]
        )
        in_ranked = any(
            tuple(p.get("super_labels") or []) == tuple(labs) for p in ctx["ranked"]
        )
        if in_scored or in_ranked:
            lines.append(
                f"- 子鏈 **有出現在** beam 葉／假設池 "
                f"(scored={in_scored}, ranked={in_ranked}) → **未被剪枝**"
            )
            # still show depth trace for transparency
        else:
            lines.append("- 子鏈未出現在 beam 葉／假設池 → 進一步追蹤前綴：")

        trace = beam_trace_gt_prefix(
            nodes, succ, calib, c, beam_width=64, dt_scoring=False, transition_prior=False
        )
        lines.append("")
        lines.append("| depth | prefix | gt_rank/n_nxt | gt_edgeΣ | cutoff@64 | 狀態 |")
        lines.append("|---:|---|---|---:|---:|---|")
        for d in trace["depths"]:
            gr = d["gt_rank_in_nxt"]
            gr_s = f"{gr}/{d['n_nxt']}" if gr is not None else f"—/{d['n_nxt']}"
            gs = f"{d['gt_edge_score']:.3f}" if d["gt_edge_score"] is not None else "—"
            cut = f"{d['cutoff_score']:.3f}" if d["cutoff_score"] is not None else "—"
            lines.append(
                f"| {d['depth']} | `{' -> '.join(d['target'])}` | {gr_s} | {gs} | {cut} | {d['reason']} |"
            )
        if in_scored:
            verdict = "存活於池中（未被剪枝）"
        elif any(d["reason"] == "被 beam 剪掉" for d in trace["depths"]):
            verdict = "被 beam 剪掉"
        elif any("無此 GT 前綴" in d["reason"] or "無任何擴展" in d["reason"] for d in trace["depths"]):
            verdict = "無合法路徑／前綴無法展開（非單純 beam 分數剪枝）"
        elif trace["alive_to_end"]:
            verdict = "beam 追蹤顯示前綴可活到終點，但未進葉集合（需查 max_leaves／起點）"
        else:
            verdict = "未進池（見上表）"
        lines.append("")
        lines.append(f"**判定：{verdict}**")
        if any(d["reason"] == "被 beam 剪掉" for d in trace["depths"]):
            lines.append("")
            lines.append(
                "註：beam 內部累加的是**邊分 only**（不含 node evidence）；"
                "故即使完工後 node+edge 總分高於 Top-1，仍可能在中途被剪。"
            )

    # 補充：完整 GT 超節點序
    lines.append("### 完整 GT 超節點序（含斷邊）")
    full = ctx["gt_super_seq"]
    lines.append("`" + " -> ".join(nodes[i].label for i in full) + "`")
    broken = []
    for a, b in zip(full, full[1:]):
        if not any(j == b for j, *_ in succ[a]):
            broken.append((nodes[a].label, nodes[b].label))
    if broken:
        lines.append("斷邊（succ 無此邊）→ 完整序 **無合法路徑**：")
        for a, b in broken:
            lines.append(f"- `{a} → {b}`")
    else:
        lines.append("完整序在 succ 上連通；若未進池則屬 beam 剪枝問題。")


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_tids = list(gt["gt_tids"])
    gt_set = set(gt_tids)
    calib = pickle.load(CALIB.open("rb"))

    tp.SIM_MIN = SIM_MIN
    tp.configure_for_input(str(MERGE))
    tp.apply_llr_emb_gates(True)
    tracks = tp.load_tracks(str(MERGE))

    lines = [
        "# 車輛 0507 診斷（純輸出）",
        "",
        f"候選={len(tracks)}  GT={len(gt_tids)}  SIM_MIN={SIM_MIN}  "
        f"calibration=`{CALIB.name}`  DT_MAX={tp.DT_MAX}",
        "",
    ]

    section_identity(tracks, gt_set, lines)
    lines.append("")
    ctx = section_gt_survival(tracks, gt_tids, calib, lines)
    lines.append("")
    section_top1_edges(ctx, gt_set, lines)
    lines.append("")
    section_beam(ctx, calib, lines)

    out_md = DIAG_DIR / "diagnose_vehicle_0507.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"寫入：{out_md}")
    # also dump compact json for ranks
    meta = {
        "n_tracks": len(tracks),
        "top1_score": ctx["top1_score"],
        "max_legal_track_chain": ctx["chains"],
        "max_legal_super_chains": [
            [ctx["nodes"][i].label for i in c] for c in ctx["best_super_chains"]
        ],
        "chain_results": [
            {
                "labs": r["labs"],
                "score": r["score"],
                "exact_rank": r["exact_rank"],
                "exact_leaf_rank": r["exact_leaf_rank"],
                "err": r["err"],
            }
            for r in ctx["chain_results"]
        ],
    }
    (DIAG_DIR / "diagnose_vehicle_0507.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE")


if __name__ == "__main__":
    main()
