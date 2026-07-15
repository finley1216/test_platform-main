# -*- coding: utf-8 -*-
"""
B 設定：精確極大路徑計分／排名、confirmed_negatives、視覺化
=========================================================
設定：dt_scoring=off, transition_prior=off, emb_gate+supernode+node_evidence=on
不經 beam：全量 leaf DFS。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.special import logsumexp

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import llr_gate_config as gates  # noqa: E402
import visualize_fixed_paths as viz  # noqa: E402
from evaluate_paths import load_gt, precision_recall  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

GT_CHAIN = [
    "K8-09_7",
    "{K8-08_30,K8-01_7}",
    "K8-07_40",
    "{K8-23_8,K8-22_19}",
    "K8-22_22",
    "K8-07_112",
    "K8-01_50",
    "{K8-08_77,K8-01_62}",
]
DIRECT = [
    "K8-09_7",
    "{K8-08_30,K8-01_7}",
    "K8-07_40",
    "{K8-23_8,K8-22_19}",
    "K8-07_112",
    "K8-01_50",
    "{K8-08_77,K8-01_62}",
]


def expand_labels(labels: list[str]) -> list[str]:
    out = []
    for lab in labels:
        if lab.startswith("{") and lab.endswith("}"):
            out.extend([x.strip() for x in lab[1:-1].split(",") if x.strip()])
        else:
            out.append(lab)
    return out


def resolve_chain_nodes(nodes: list, labels: list[str]):
    by_member = {}
    for sn in nodes:
        for tid in sn.tids:
            by_member[tid] = sn
    chain = []
    seen = set()
    for lab in labels:
        if lab.startswith("{"):
            mems = [x.strip() for x in lab[1:-1].split(",") if x.strip()]
            sn = by_member[mems[0]]
        else:
            sn = by_member[lab]
        if sn.sid in seen:
            continue
        seen.add(sn.sid)
        chain.append(sn)
    return chain


def score_labeled_path(nodes, labels, calib, *, dt_scoring: bool, transition_prior: bool):
    """精確逐邊+節點計分（含 hist gate）。失敗回傳 ok=False。"""
    chain = resolve_chain_nodes(nodes, labels)
    edges = []
    hist = [chain[0].emb]
    for i in range(len(chain) - 1):
        sa, sb = chain[i], chain[i + 1]
        best, _ = llr._best_member_edge(sa, sb)
        if best is None:
            return {"ok": False, "reason": f"no edge {sa.label}->{sb.label}"}
        u, v, dt, hop, emb, h_dist = best
        ok_h, hsim, need = llr._hist_ok(hist, chain, i + 1, h_dist)
        if not ok_h:
            return {
                "ok": False,
                "reason": f"hist fail {sa.label}->{sb.label} hist={hsim:.3f}<{need}",
            }
        e = llr.edge_llr(
            u,
            v,
            dt,
            emb,
            h_dist,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
        )
        e["hop"] = hop
        e["hist_emb"] = hsim
        e["from_super"] = sa.label
        e["to_super"] = sb.label
        e["from_members"] = sa.tids
        e["to_members"] = sb.tids
        e["via"] = f"{u.tid}->{v.tid}"
        edges.append(e)
        hist.append(sb.emb)
    score, node_ev = llr.path_score_llr(chain, edges, calib)
    tids = []
    for sn in chain:
        tids.extend(sn.tids)
    return {
        "ok": True,
        "score": score,
        "tids": tids,
        "super_labels": [sn.label for sn in chain],
        "edges": edges,
        "node_evidence": node_ev,
    }


def enumerate_maximal_exact(
    tracks,
    calib,
    *,
    dt_scoring: bool = False,
    transition_prior: bool = False,
):
    """全量 leaf DFS（不經 beam），回傳極大路徑 list（含節點證據總分）。"""
    nodes, srep = llr.build_supernodes(tracks)
    succ, _, n_legal = llr._build_succ(nodes)
    # 預算邊 LLR（加速）
    edge_cache = {}
    for i in range(len(nodes)):
        for j, u, v, dt, hop, emb, h_dist in succ[i]:
            e = llr.edge_llr(
                u,
                v,
                dt,
                emb,
                h_dist,
                calib,
                dt_scoring=dt_scoring,
                transition_prior=transition_prior,
            )
            e["hop"] = hop
            e["from_super"] = nodes[i].label
            e["to_super"] = nodes[j].label
            e["from_members"] = nodes[i].tids
            e["to_members"] = nodes[j].tids
            e["via"] = f"{u.tid}->{v.tid}"
            edge_cache[(i, j)] = (e, h_dist)

    # 節點分數預算
    node_scores = []
    node_evs_tmpl = []
    for sn in nodes:
        ne = llr.node_evidence(sn.sim, calib)
        node_scores.append(ne["score"])
        node_evs_tmpl.append({"super": sn.label, "members": sn.tids, **ne})

    maximal = []
    t0 = time.time()

    def dfs(idx, path_idx, edges_info, hist_embs, edge_sum, node_sum):
        extended = False
        for j, u, v, dt, hop, emb, h_dist in succ[idx]:
            if j in path_idx:
                continue
            ok_h, hsim, _ = llr._hist_ok(hist_embs, nodes, j, h_dist)
            if not ok_h:
                continue
            e0, _ = edge_cache[(idx, j)]
            e = dict(e0)
            e["hist_emb"] = hsim
            extended = True
            path_idx.append(j)
            edges_info.append(e)
            hist_embs.append(nodes[j].emb)
            dfs(
                j,
                path_idx,
                edges_info,
                hist_embs,
                edge_sum + e["score"],
                node_sum + node_scores[j],
            )
            hist_embs.pop()
            path_idx.pop()
            edges_info.pop()
        if not extended:
            tids = []
            for i in path_idx:
                tids.extend(nodes[i].tids)
            maximal.append(
                {
                    "tids": tids,
                    "super_labels": [nodes[i].label for i in path_idx],
                    "super_ids": [nodes[i].sid for i in path_idx],
                    "score": edge_sum + node_sum,
                    "edges": list(edges_info),
                    "node_evidence": [node_evs_tmpl[i] for i in path_idx],
                }
            )

    for s in range(len(nodes)):
        dfs(s, [s], [], [nodes[s].emb], 0.0, node_scores[s])

    maximal.sort(key=lambda p: -p["score"])
    # softmax
    if maximal:
        scores = np.asarray([p["score"] for p in maximal], dtype=np.float64)
        log_z = logsumexp(scores)
        for p, s in zip(maximal, scores):
            p["path_probability"] = float(math.exp(s - log_z))
    elapsed = time.time() - t0
    return maximal, nodes, srep, n_legal, elapsed


def rank_of_labels(maximal, labels):
    key_labs = tuple(labels)
    key_tids = tuple(expand_labels(labels))
    for i, p in enumerate(maximal, 1):
        if tuple(p.get("super_labels") or []) == key_labs:
            return i, p
    for i, p in enumerate(maximal, 1):
        if tuple(p["tids"]) == key_tids:
            return i, p
    return None, None


def render_review_09_42(merge_dir: Path, by_tid: dict, out_png: Path, gt_path: Path):
    tid = "K8-09_42"
    meta = viz.track_meta(by_tid, tid)
    crops = viz.load_crops(merge_dir, tid)
    three = viz._pick_three(crops)
    thumb = (160, 220)
    margin = 20
    title_h = 70
    foot_h = 80
    cell_w = thumb[0] + 24
    width = margin * 2 + 3 * cell_w + 40
    height = title_h + thumb[1] + foot_h + 100
    img = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    font_t = viz._font(16)
    font_s = viz._font(12)
    font_xs = viz._font(11)
    draw.rectangle([8, 8, width - 9, height - 9], outline=(200, 40, 40), width=4)
    draw.text((margin, 14), "confirmed_negatives: K8-09_42", fill=(180, 0, 0), font=font_t)
    draw.text(
        (margin, 36),
        "僅供評估／報告標注 — 不進硬規則或計分",
        fill=(80, 80, 80),
        font=font_xs,
    )
    for i, cp in enumerate(three):
        x = margin + i * (cell_w + 10)
        y = title_h
        draw.rectangle([x, y, x + cell_w - 1, y + thumb[1] + 50], outline=(200, 40, 40), width=2)
        label = ["first", "mid", "last"][i]
        draw.text((x + 6, y + 4), label, fill=(0, 0, 0), font=font_s)
        if cp is not None and cp.is_file():
            img.paste(viz._thumb(cp, thumb), (x + 12, y + 22))
        else:
            draw.text((x + 20, y + 80), "no crop", fill=(160, 0, 0), font=font_s)
    ts, te = meta["t_start"], meta["t_end"]
    sim = meta["sim"]
    draw.text(
        (margin, title_h + thumb[1] + 60),
        f"{tid}  cam={meta['cam']}  sim={sim:.3f}  [{ts:.2f}–{te:.2f}]",
        fill=(20, 20, 20),
        font=font_s,
    )
    draw.text(
        (margin, title_h + thumb[1] + 80),
        f"source: {gt_path.name} → confirmed_negatives",
        fill=(100, 100, 100),
        font=font_xs,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def update_gt_json(gt_path: Path) -> dict:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    data["confirmed_negatives"] = {
        "K8-09_42": {
            "status": "confirmed_non_target",
            "note": "僅供評估與報告標注，不進入硬規則／候選篩選／LLR 計分或校準樣本。",
            "annotated_at": "2026-07-15",
            "reason": "目視確認非 query 目標；B 設定 Top-1 繞路節點",
        }
    }
    # 相容舊語意：其餘非 GT 仍可當評估用，但不強制列出
    data["note"] = (
        "K8-08_43 已自 GT 剔除（誤標）。K8-09_42 列入 confirmed_negatives（僅標注）。"
        "GT／confirmed_negatives 均不進入硬規則或計分。"
    )
    gt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def render_timeline_with_negatives(
    paths,
    by_tid,
    out_png,
    *,
    title: str,
    confirmed_negatives: set[str],
):
    """時間軸：一般路徑色塊；confirmed_negatives 強制紅色標注。"""
    # reuse viz.render_timeline then overlay? clearer to fork with red for negatives
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
    ]
    COLOR_NEG = (220, 20, 20)
    all_tids = []
    for p in paths:
        all_tids.extend(p.get("tids") or [])
    metas = [viz.track_meta(by_tid, t) for t in all_tids if t in by_tid]
    if not metas:
        return viz.render_timeline(paths, by_tid, out_png, title=title)

    cams = sorted({m["cam"] for m in metas})
    t_min = min(m["t_start"] for m in metas)
    t_max = max(m["t_end"] for m in metas)
    if t_max <= t_min:
        t_max = t_min + 1.0

    margin_l, margin_r, margin_t, margin_b = 90, 20, 56, 56
    row_h = 36
    width = 1100
    height = margin_t + margin_b + len(cams) * row_h + 18 * len(paths)
    img = Image.new("RGB", (width, height), viz.BG)
    draw = ImageDraw.Draw(img)
    font_xs = viz._font(10)
    draw.text((16, 10), title, fill=(20, 20, 20), font=viz._font(15))
    draw.text(
        (16, 32),
        "red block/label = confirmed_negatives (e.g. K8-09_42)",
        fill=COLOR_NEG,
        font=font_xs,
    )
    plot_w = width - margin_l - margin_r
    plot_h0 = margin_t

    def x_of(t: float) -> int:
        return margin_l + int((t - t_min) / (t_max - t_min) * plot_w)

    for yi, cam in enumerate(cams):
        y = plot_h0 + yi * row_h
        draw.text((8, y + 8), cam, fill=(30, 30, 30), font=font_xs)
        draw.line([(margin_l, y + row_h - 1), (width - margin_r, y + row_h - 1)], fill=(220, 220, 220))

    for pi, p in enumerate(paths):
        color = palette[pi % len(palette)]
        cols = viz.parse_super_columns(p)
        label = f"#{pi+1}"
        if p.get("path_probability") is not None:
            label += f" P={p['path_probability']:.3f}"
        draw.text((margin_l, height - margin_b + 6 + pi * 14), label, fill=color, font=font_xs)

        pts = []
        for col in cols:
            for tid in col["members"]:
                m = viz.track_meta(by_tid, tid)
                if m["t_start"] is None:
                    continue
                yi = cams.index(m["cam"])
                y = plot_h0 + yi * row_h + 6 + pi * 3
                x0 = x_of(m["t_start"])
                x1 = x_of(max(m["t_end"], m["t_start"] + 0.3))
                if x1 <= x0:
                    x1 = x0 + 4
                y1 = y + 10
                fill = COLOR_NEG if tid in confirmed_negatives else color
                for yy in range(y, y1 + 1):
                    draw.line([(x0, yy), (x1, yy)], fill=fill)
                short = tid.split("_", 1)[-1]
                if tid in confirmed_negatives:
                    short = f"{short}!"
                draw.text((x0 + 2, y - 1), short, fill=(255, 255, 255), font=font_xs)
            tid0 = col["members"][0]
            m0 = viz.track_meta(by_tid, tid0)
            if m0["t_start"] is None:
                continue
            yi = cams.index(m0["cam"])
            y = plot_h0 + yi * row_h + 10 + pi * 3
            pts.append((x_of((m0["t_start"] + m0["t_end"]) / 2), y))
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=color, width=2)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = t_min + frac * (t_max - t_min)
        xx = x_of(t)
        draw.line([(xx, plot_h0 - 4), (xx, plot_h0 + len(cams) * row_h)], fill=(200, 200, 200))
        draw.text((xx - 10, plot_h0 + len(cams) * row_h + 2), f"{t:.0f}s", fill=(80, 80, 80), font=font_xs)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def main():
    merge_dir = (QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507").resolve()
    out_root = (OUTPUT_ROOT / "path_enum_llr").resolve()
    fixed_dir = out_root / "gt_calib_0507_fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_root / "ground_truth_20260507.json"
    calib = llr.load_calibration(out_root / "calibration_gt0507.pkl")

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge_dir))
    gates.apply_llr_emb_gates(True)
    tracks = pes.load_tracks(str(merge_dir))
    by_tid = {t.tid: t for t in tracks}

    # --- 2. confirmed_negatives ---
    gt_data = update_gt_json(gt_path)
    neg = set((gt_data.get("confirmed_negatives") or {}).keys())
    review_png = fixed_dir / "review_09_42.png"
    render_review_09_42(merge_dir, by_tid, review_png, gt_path)
    print(f"wrote {review_png}")
    print(f"updated {gt_path} confirmed_negatives={list(neg)}")

    # --- 1. exact maximal under B ---
    print("exact maximal DFS (B: dt off, prior off)…")
    maximal, nodes, srep, n_legal, elapsed = enumerate_maximal_exact(
        tracks, calib, dt_scoring=False, transition_prior=False
    )
    print(f"n_legal_edges={n_legal}  n_maximal={len(maximal)}  elapsed={elapsed:.1f}s")

    # Softmax attach already done; also score labeled paths for edge tables
    gt_exact = score_labeled_path(
        nodes, GT_CHAIN, calib, dt_scoring=False, transition_prior=False
    )
    di_exact = score_labeled_path(
        nodes, DIRECT, calib, dt_scoring=False, transition_prior=False
    )
    r_gt, p_gt = rank_of_labels(maximal, GT_CHAIN)
    r_di, p_di = rank_of_labels(maximal, DIRECT)

    # Top-1..3 from exact
    gt_set = set(gt_data["person_tids"])
    top3 = []
    for i, p in enumerate(maximal[:3], 1):
        pr = precision_recall(p["tids"], gt_set)
        top3.append(
            {
                "rank": i,
                "precision": pr["precision"],
                "recall": pr["recall"],
                "score": p["score"],
                "path_probability": p.get("path_probability"),
                "path": " -> ".join(p.get("super_labels") or p["tids"]),
                "tids": p["tids"],
                "super_labels": p.get("super_labels"),
                "edges": p.get("edges"),
                "node_evidence": p.get("node_evidence"),
            }
        )

    report = {
        "setting": "B: dt_scoring=off, transition_prior=off, emb_gate+supernode+node_ev=on",
        "enumeration": "full leaf DFS (no beam)",
        "n_legal_edges": n_legal,
        "n_maximal": len(maximal),
        "elapsed_sec": elapsed,
        "top3": [
            {
                "rank": t["rank"],
                "precision": t["precision"],
                "recall": t["recall"],
                "score": t["score"],
                "path_probability": t["path_probability"],
                "path": t["path"],
            }
            for t in top3
        ],
        "gt_chain_exact": {
            **{k: gt_exact[k] for k in ("ok", "score", "super_labels", "tids", "reason") if k in gt_exact},
            "rank_among_maximal": r_gt,
            "path_probability": p_gt.get("path_probability") if p_gt else None,
            "edge_scores": [
                {
                    "from": e.get("from_super"),
                    "to": e.get("to_super"),
                    "via": e.get("via"),
                    "dt": e.get("dt"),
                    "LLR_dt": e.get("LLR_dt"),
                    "LLR_emb": e.get("LLR_emb"),
                    "score": e.get("score"),
                }
                for e in (gt_exact.get("edges") or [])
            ],
            "node_scores": [
                {"super": n.get("super"), "score": n.get("score"), "sim": n.get("sim")}
                for n in (gt_exact.get("node_evidence") or [])
            ],
        },
        "direct_exact": {
            **{k: di_exact[k] for k in ("ok", "score", "super_labels", "tids", "reason") if k in di_exact},
            "rank_among_maximal": r_di,
            "path_probability": p_di.get("path_probability") if p_di else None,
            "edge_scores": [
                {
                    "from": e.get("from_super"),
                    "to": e.get("to_super"),
                    "via": e.get("via"),
                    "dt": e.get("dt"),
                    "LLR_dt": e.get("LLR_dt"),
                    "LLR_emb": e.get("LLR_emb"),
                    "score": e.get("score"),
                }
                for e in (di_exact.get("edges") or [])
            ],
            "node_scores": [
                {"super": n.get("super"), "score": n.get("score"), "sim": n.get("sim")}
                for n in (di_exact.get("node_evidence") or [])
            ],
        },
        "confirmed_negatives": gt_data.get("confirmed_negatives"),
        "review_png": str(review_png),
    }

    # --- 3. visualizations from B exact top ---
    viz_paths = maximal[:10]
    out_files = []
    p1 = fixed_dir / f"{merge_dir.name}_top1_collage.png"
    viz.render_path_collage(
        merge_dir,
        viz_paths[0],
        by_tid,
        p1,
        title="Top-1 (B: dt-scoring=off)",
        gt_set=gt_set,
        highlight_tid="K8-09_42",
        calib=calib,
    )
    out_files.append(p1)
    collage_map = {1: p1.name}
    for rank in (2, 3):
        if rank - 1 < len(viz_paths):
            outp = fixed_dir / f"{merge_dir.name}_top{rank}_collage.png"
            # for direct path (likely top2), don't amplify 09_42 if absent
            hi = "K8-09_42" if "K8-09_42" in viz_paths[rank - 1]["tids"] else None
            viz.render_path_collage(
                merge_dir,
                viz_paths[rank - 1],
                by_tid,
                outp,
                title=f"Top-{rank} (B: dt-scoring=off)",
                gt_set=gt_set,
                highlight_tid=hi,
                calib=calib,
            )
            out_files.append(outp)
            collage_map[rank] = outp.name

    tl = fixed_dir / f"{merge_dir.name}_top3_timeline.png"
    render_timeline_with_negatives(
        viz_paths[:3],
        by_tid,
        tl,
        title=f"{merge_dir.name} Top-3 timeline (B) — 09_42 red=confirmed_neg",
        confirmed_negatives=neg,
    )
    out_files.append(tl)

    # GT chain collage
    if gt_exact.get("ok"):
        gt_png = fixed_dir / f"{merge_dir.name}_gt_chain_collage.png"
        viz.render_path_collage(
            merge_dir,
            gt_exact,
            by_tid,
            gt_png,
            title=f"GT 11/11 exact (B) rank=#{r_gt}",
            gt_set=gt_set,
            highlight_tid=None,
            calib=calib,
        )
        out_files.append(gt_png)

    # refresh top1 json for B
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "person",
        "scoring": "LLR",
        "options": {
            "use_emb_gate_fix": True,
            "use_supernode": True,
            "use_node_evidence": True,
            "dt_scoring": False,
            "transition_prior": False,
            "enumeration": "full_leaf_dfs",
        },
        "confirmed_negatives": gt_data.get("confirmed_negatives"),
        "n_paths_maximal": len(maximal),
        "n_legal_edges": n_legal,
        "top1": top3[0] if top3 else None,
        "top10_paths": [
            {
                "rank": i,
                "score": p["score"],
                "path_probability": p.get("path_probability"),
                "tids": p["tids"],
                "super_labels": p.get("super_labels"),
            }
            for i, p in enumerate(maximal[:10], 1)
        ],
        "exact_ranks": {
            "gt_chain": report["gt_chain_exact"],
            "direct": report["direct_exact"],
        },
    }
    (fixed_dir / f"{merge_dir.name}_llr_top1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    out_json = fixed_dir / "b_exact_ranks_0507.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out_files.append(out_json)
    out_files.append(review_png)

    print("\n=== B exact ranks ===")
    print(f"GT chain: ok={gt_exact.get('ok')} score={gt_exact.get('score')} rank=#{r_gt} P={report['gt_chain_exact'].get('path_probability')}")
    print(f"Direct:   ok={di_exact.get('ok')} score={di_exact.get('score')} rank=#{r_di} P={report['direct_exact'].get('path_probability')}")
    if maximal:
        print(f"Top-1: score={maximal[0]['score']:.4f} P={maximal[0]['path_probability']:.4f}  "
              + " -> ".join(maximal[0].get("super_labels") or []))
    print("\n=== files ===")
    for f in out_files:
        print(Path(f).resolve())


if __name__ == "__main__":
    main()
