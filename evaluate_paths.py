# -*- coding: utf-8 -*-
"""
路徑評估 + GT 軌跡可行性診斷
================================
GT 只用於評估／診斷，不進入硬規則或候選篩選。

用法：
  python3 evaluate_paths.py \\
    --gt ../output/path_enum_llr/ground_truth_20260507.json \\
    --old-json ../output/path_enum/人員追蹤_20260507_top1.json \\
    --llr-json ../output/path_enum_llr/人員追蹤_20260507_llr_top1.json \\
    --merge-dir ../output/query_filter_merge/人員追蹤_20260507 \\
    --sim-min 0.85 \\
    --out-dir ../output/path_enum_llr
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

N_GT = 11  # K8-08_43 已自 GT 剔除（誤標）
SPECIAL_TIDS = ("K8-07_112", "K8-22_22")


def load_gt(path: Path) -> dict:
    gt = json.loads(path.read_text(encoding="utf-8"))
    tids = list(gt["person_tids"])
    if len(tids) != N_GT:
        print(f"警告：GT 標注 {len(tids)} 條（預期 {N_GT}）")
    return gt


def precision_recall(path_tids: list[str], gt_set: set[str]) -> dict:
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    return {
        "n_path": n,
        "n_hit": hit,
        "precision": (hit / n) if n else 0.0,
        "recall": hit / float(N_GT),
        "hit_tids": [t for t in path_tids if t in gt_set],
        "miss_tids": [t for t in path_tids if t not in gt_set],
    }


def eval_top_json(top_json: Path, gt_set: set[str]) -> dict:
    data = json.loads(top_json.read_text(encoding="utf-8"))
    rows = []
    for p in data.get("top10_paths") or []:
        tids = p["tids"]
        pr = precision_recall(tids, gt_set)
        rows.append(
            {
                "rank": p.get("rank"),
                "score": p.get("score"),
                "path_probability": p.get("path_probability"),
                "path": " -> ".join(tids),
                "tids": tids,
                **pr,
            }
        )
    return {
        "json": str(top_json.resolve()),
        "n_paths_all": data.get("n_paths_all"),
        "n_paths_maximal": data.get("n_paths_maximal"),
        "top10": rows,
    }


def _score_all_paths_old(tracks: list) -> list[dict]:
    all_paths, _ = pes.enumerate_paths(tracks)
    scored = []
    for path_idx, edges_info in all_paths:
        path = [tracks[i] for i in path_idx]
        scored.append(
            {
                "tids": [t.tid for t in path],
                "score": pes.path_score(path, edges_info),
                "edges": edges_info,
            }
        )
    scored.sort(key=lambda p: -p["score"])
    return scored


def _score_all_paths_llr(tracks: list, calib: dict) -> list[dict]:
    all_paths, _, _ = llr.enumerate_paths_llr(tracks, calib)
    scored = []
    for path_idx, edges_info in all_paths:
        path = [tracks[i] for i in path_idx]
        scored.append(
            {
                "tids": [t.tid for t in path],
                "score": llr.path_score_llr(path, edges_info),
                "edges": edges_info,
            }
        )
    scored.sort(key=lambda p: -p["score"])
    return scored


def find_gt_best_path(scored_paths: list[dict], gt_set: set[str]) -> dict | None:
    """在所有枚舉路徑中找 precision=100% 且 recall 最高者；同分取較短／較早出現。"""
    best = None
    for rank, p in enumerate(scored_paths, 1):
        # rank here is within this scored list (already sorted by that method's score)
        pr = precision_recall(p["tids"], gt_set)
        if pr["precision"] < 1.0 - 1e-12:
            continue
        cand = {
            "tids": p["tids"],
            "path": " -> ".join(p["tids"]),
            "score": p["score"],
            "rank_in_scored": rank,
            **pr,
        }
        if best is None:
            best = cand
            continue
        if cand["recall"] > best["recall"]:
            best = cand
        elif abs(cand["recall"] - best["recall"]) < 1e-12 and cand["n_path"] < best["n_path"]:
            best = cand
    return best


def rank_of_path(scored_paths: list[dict], tids: list[str]) -> int | None:
    key = tuple(tids)
    for i, p in enumerate(scored_paths, 1):
        if tuple(p["tids"]) == key:
            return i
    return None


def _time_overlap(a: pes.Track, b: pes.Track) -> bool:
    return not (a.t_end < b.t_start or b.t_end < a.t_start)


def diagnose_gt_feasibility(tracks: list, gt_tids: list[str]) -> dict:
    by_tid = {t.tid: t for t in tracks}
    missing = [t for t in gt_tids if t not in by_tid]
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    gt_tracks_sorted = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))

    overlaps = []
    for i, a in enumerate(gt_tracks_sorted):
        for b in gt_tracks_sorted[i + 1 :]:
            if _time_overlap(a, b):
                overlaps.append(
                    {
                        "a": a.tid,
                        "b": b.tid,
                        "a_cam": a.cam,
                        "b_cam": b.cam,
                        "a_span": [a.t_start, a.t_end],
                        "b_span": [b.t_start, b.t_end],
                        "overlap_sec": min(a.t_end, b.t_end) - max(a.t_start, b.t_start),
                    }
                )

    # 時間相鄰（排序後 consecutive）edge_check
    consecutive = []
    for u, v in zip(gt_tracks_sorted, gt_tracks_sorted[1:]):
        ok, reason, dt, hop, emb, h_dist = pes.edge_check(u, v)
        consecutive.append(
            {
                "from": u.tid,
                "to": v.tid,
                "ok": bool(ok),
                "reason": reason or "",
                "dt": float(dt) if dt is not None else None,
                "hop": hop,
                "emb": float(emb) if emb is not None else None,
                "h_dist": float(h_dist) if h_dist is not None else None,
                "u_span": [u.t_start, u.t_end],
                "v_span": [v.t_start, v.t_end],
            }
        )

    # 特殊三條：與所有其他 GT 的雙向 edge_check
    special = {}
    for tid in SPECIAL_TIDS:
        if tid not in by_tid:
            special[tid] = {"missing_in_candidates": True, "as_from": [], "as_to": []}
            continue
        u = by_tid[tid]
        as_from, as_to = [], []
        for v in gt_tracks:
            if v.tid == tid:
                continue
            ok, reason, dt, hop, emb, h_dist = pes.edge_check(u, v)
            as_from.append(
                {
                    "to": v.tid,
                    "ok": bool(ok),
                    "reason": reason or "",
                    "dt": float(dt) if dt is not None else None,
                    "hop": hop,
                    "emb": float(emb) if emb is not None else None,
                    "h_dist": float(h_dist) if h_dist is not None else None,
                }
            )
            ok2, reason2, dt2, hop2, emb2, h_dist2 = pes.edge_check(v, u)
            as_to.append(
                {
                    "from": v.tid,
                    "ok": bool(ok2),
                    "reason": reason2 or "",
                    "dt": float(dt2) if dt2 is not None else None,
                    "hop": hop2,
                    "emb": float(emb2) if emb2 is not None else None,
                    "h_dist": float(h_dist2) if h_dist2 is not None else None,
                }
            )
        special[tid] = {
            "missing_in_candidates": False,
            "span": [u.t_start, u.t_end],
            "sim": u.sim,
            "n_ok_as_from": sum(1 for x in as_from if x["ok"]),
            "n_ok_as_to": sum(1 for x in as_to if x["ok"]),
            "as_from": as_from,
            "as_to": as_to,
        }

    # GT 子圖：所有有序對的 edge_check；再用 DFS + EMB_HIST_MIN 找最長可行路徑
    gt_list = gt_tracks_sorted
    n = len(gt_list)
    succ = [[] for _ in range(n)]
    all_edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u, v = gt_list[i], gt_list[j]
            ok, reason, dt, hop, emb, h_dist = pes.edge_check(u, v)
            rec = {
                "from": u.tid,
                "to": v.tid,
                "ok": bool(ok),
                "reason": reason or "",
                "dt": float(dt) if dt is not None else None,
                "hop": hop,
                "emb": float(emb) if emb is not None else None,
                "h_dist": float(h_dist) if h_dist is not None else None,
            }
            all_edges.append(rec)
            if ok:
                succ[i].append((j, dt, hop, emb, h_dist))

    longest = {"tids": [], "n": 0, "edges": []}

    def dfs(idx, path_idx, edges_info, hist_embs):
        nonlocal longest
        tids = [gt_list[k].tid for k in path_idx]
        if len(tids) > longest["n"]:
            longest = {
                "tids": list(tids),
                "n": len(tids),
                "edges": list(edges_info),
            }
        for j, dt, hop, emb, h_dist in succ[idx]:
            if j in path_idx:
                continue
            v = gt_list[j]
            hsim = pes.hist_emb_sim(hist_embs, v)
            emb_need = (
                pes.EMB_HIST_MIN - 0.02
                if (h_dist is not None and h_dist < pes.H_DIST_GATE)
                else pes.EMB_HIST_MIN
            )
            if hsim < emb_need:
                continue
            u = gt_list[idx]
            edges_info.append(
                {
                    "from": u.tid,
                    "to": v.tid,
                    "dt": dt,
                    "hop": hop,
                    "emb": emb,
                    "hist_emb": hsim,
                    "h_dist": h_dist,
                }
            )
            path_idx.append(j)
            hist_embs.append(v.emb)
            dfs(j, path_idx, edges_info, hist_embs)
            hist_embs.pop()
            path_idx.pop()
            edges_info.pop()

    for s in range(n):
        dfs(s, [s], [], [gt_list[s].emb])

    # 瓶頸：連續時間序中被拒的邊，以及最長路徑未覆蓋的 GT
    bottlenecks = [e for e in consecutive if not e["ok"]]
    uncovered = [t for t in gt_tids if t not in set(longest["tids"])]

    return {
        "n_gt": len(gt_tids),
        "n_gt_in_candidates": len(gt_tracks),
        "missing_from_candidates": missing,
        "gt_sorted_by_t_start": [
            {
                "tid": t.tid,
                "cam": t.cam,
                "t_start": t.t_start,
                "t_end": t.t_end,
                "sim": t.sim,
            }
            for t in gt_tracks_sorted
        ],
        "time_overlaps_among_gt": overlaps,
        "consecutive_edge_checks": consecutive,
        "special_tracks": special,
        "n_legal_gt_edges": sum(1 for e in all_edges if e["ok"]),
        "n_checked_gt_ordered_pairs": len(all_edges),
        "longest_feasible_path": longest,
        "max_gt_coverable": longest["n"],
        "uncovered_by_longest": uncovered,
        "bottleneck_consecutive_edges": bottlenecks,
        "note": (
            "時間重疊的 GT 對如實列出，不当成異常剔除。"
            "最長可行路徑使用與管線相同的 edge_check + EMB_HIST_MIN gate。"
        ),
    }


def write_diagnose_txt(diag: dict, out_txt: Path) -> None:
    lines = []
    lines.append("=== GT 軌跡可行性診斷 ===")
    lines.append(f"GT 數={diag['n_gt']}  候選中找到={diag['n_gt_in_candidates']}")
    if diag["missing_from_candidates"]:
        lines.append("缺失：" + ", ".join(diag["missing_from_candidates"]))
    lines.append("")
    lines.append("--- GT 按 t_start 排序 ---")
    for r in diag["gt_sorted_by_t_start"]:
        lines.append(
            f"  {r['tid']:12} cam={r['cam']}  "
            f"[{r['t_start']:.1f}, {r['t_end']:.1f}]  sim={r['sim']:.3f}"
        )
    lines.append("")
    lines.append(f"--- GT 之間時間重疊（共 {len(diag['time_overlaps_among_gt'])} 對）---")
    if not diag["time_overlaps_among_gt"]:
        lines.append("  （無）")
    for o in diag["time_overlaps_among_gt"]:
        lines.append(
            f"  {o['a']} ↔ {o['b']}  overlap={o['overlap_sec']:.2f}s  "
            f"cams={o['a_cam']}|{o['b_cam']}"
        )
    lines.append("")
    lines.append("--- 時間相鄰 GT 邊（edge_check）---")
    for e in diag["consecutive_edge_checks"]:
        if e["ok"]:
            h = f"{e['h_dist']:.1f}px" if e["h_dist"] is not None else "—"
            lines.append(
                f"  OK   {e['from']} -> {e['to']}  "
                f"hop={e['hop']} dt={e['dt']:.2f}s emb={e['emb']:.3f} h={h}"
            )
        else:
            lines.append(f"  REJECT {e['from']} -> {e['to']}  reason={e['reason']}")
    lines.append("")
    lines.append("--- 特殊 track（K8-07_112 / K8-22_22；08_43 已自 GT 剔除）---")
    for tid in SPECIAL_TIDS:
        sp = diag["special_tracks"].get(tid, {})
        lines.append(f"## {tid}")
        if sp.get("missing_in_candidates"):
            lines.append("  （不在候選池）")
            continue
        lines.append(
            f"  span=[{sp['span'][0]:.1f},{sp['span'][1]:.1f}] sim={sp['sim']:.3f}  "
            f"ok_as_from={sp['n_ok_as_from']} ok_as_to={sp['n_ok_as_to']}"
        )
        lines.append("  作為 from（到其他 GT）：")
        for x in sp["as_from"]:
            tag = "OK" if x["ok"] else "REJECT"
            extra = (
                f"hop={x['hop']} dt={x['dt']:.2f} emb={x['emb']:.3f}"
                if x["ok"]
                else x["reason"]
            )
            lines.append(f"    {tag:6} -> {x['to']:12}  {extra}")
        lines.append("  作為 to（從其他 GT 來）：")
        for x in sp["as_to"]:
            tag = "OK" if x["ok"] else "REJECT"
            extra = (
                f"hop={x['hop']} dt={x['dt']:.2f} emb={x['emb']:.3f}"
                if x["ok"]
                else x["reason"]
            )
            lines.append(f"    {tag:6} {x['from']:12} ->  {extra}")
    lines.append("")
    lines.append("=== 結論 ===")
    lines.append(
        f"現有硬規則（edge_check + EMB_HIST_MIN）下，"
        f"一條路徑最多能涵蓋 {diag['max_gt_coverable']} / {diag['n_gt']} 條 GT。"
    )
    lines.append("最長可行路徑：" + " -> ".join(diag["longest_feasible_path"]["tids"]))
    if diag["uncovered_by_longest"]:
        lines.append("未覆蓋：" + ", ".join(diag["uncovered_by_longest"]))
    lines.append("時間相鄰瓶頸邊：")
    if not diag["bottleneck_consecutive_edges"]:
        lines.append("  （無——連續序全部合法；瓶頸可能在非相鄰跳接或 hist gate）")
    for e in diag["bottleneck_consecutive_edges"]:
        lines.append(f"  {e['from']} -> {e['to']}: {e['reason']}")
    lines.append("")
    lines.append(diag["note"])
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate paths vs GT + diagnose GT feasibility")
    p.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "ground_truth_20260507.json",
    )
    p.add_argument(
        "--old-json",
        type=Path,
        default=OUTPUT_ROOT / "path_enum" / "人員追蹤_20260507_top1.json",
    )
    p.add_argument(
        "--llr-json",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "人員追蹤_20260507_llr_top1.json",
    )
    p.add_argument(
        "--llr-gt-json",
        type=Path,
        default=None,
        help="可選：GT 校準後 LLR top JSON",
    )
    p.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "calibration.pkl",
        help="用於重算全路徑 LLR 排名的校準檔（先驗版）",
    )
    p.add_argument(
        "--calibration-gt",
        type=Path,
        default=None,
        help="可選：GT 校準檔，用於第三套排名",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr",
    )
    args = p.parse_args(argv)

    gt = load_gt(args.gt)
    gt_set = set(gt["person_tids"])
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Top-10 評估（來自已存 JSON）---
    results = {
        "gt": gt,
        "old": eval_top_json(args.old_json, gt_set) if args.old_json.is_file() else None,
        "llr_prior": eval_top_json(args.llr_json, gt_set) if args.llr_json.is_file() else None,
        "llr_gt": None,
    }
    if args.llr_gt_json and args.llr_gt_json.is_file():
        results["llr_gt"] = eval_top_json(args.llr_gt_json, gt_set)

    # --- 全路徑掃描：GT 最佳 + 各計分排名 ---
    merge_dir = args.merge_dir.resolve()
    pes.SIM_MIN = float(args.sim_min)
    pes.configure_for_input(str(merge_dir))
    tracks = pes.load_tracks(str(merge_dir))
    all_tids = {t.tid for t in tracks}
    non_gt = sorted(all_tids - gt_set)
    results["candidate_pool"] = {
        "n_tracks": len(tracks),
        "n_gt_in_pool": sum(1 for t in gt_set if t in all_tids),
        "n_non_gt": len(non_gt),
        "non_gt_tids": non_gt,
    }

    print("枚舉並用舊法計分全部路徑…")
    scored_old = _score_all_paths_old(tracks)
    gt_best = find_gt_best_path(scored_old, gt_set)
    # GT-best 定義獨立於計分（precision=1 最大 recall）；在各套計分下找排名
    results["gt_best_path"] = None
    if gt_best:
        results["gt_best_path"] = {
            "tids": gt_best["tids"],
            "path": gt_best["path"],
            "precision": gt_best["precision"],
            "recall": gt_best["recall"],
            "n_hit": gt_best["n_hit"],
            "n_path": gt_best["n_path"],
            "rank_old": rank_of_path(scored_old, gt_best["tids"]),
            "rank_llr_prior": None,
            "rank_llr_gt": None,
            "score_old": next(
                (p["score"] for p in scored_old if p["tids"] == gt_best["tids"]), None
            ),
        }
        print(
            f"GT 最佳路徑 recall={gt_best['recall']:.3f}  "
            f"len={gt_best['n_path']}  old_rank={results['gt_best_path']['rank_old']}"
        )
        print("  " + gt_best["path"])

    if args.calibration.is_file():
        print(f"用先驗校準重算 LLR 全路徑：{args.calibration}")
        calib = llr.load_calibration(args.calibration)
        scored_llr = _score_all_paths_llr(tracks, calib)
        if results["gt_best_path"]:
            results["gt_best_path"]["rank_llr_prior"] = rank_of_path(
                scored_llr, results["gt_best_path"]["tids"]
            )
            results["gt_best_path"]["score_llr_prior"] = next(
                (
                    p["score"]
                    for p in scored_llr
                    if p["tids"] == results["gt_best_path"]["tids"]
                ),
                None,
            )
    else:
        print(f"找不到先驗校準檔，跳過 LLR 排名：{args.calibration}")

    if args.calibration_gt and args.calibration_gt.is_file():
        print(f"用 GT 校準重算 LLR 全路徑：{args.calibration_gt}")
        calib_gt = llr.load_calibration(args.calibration_gt)
        scored_llr_gt = _score_all_paths_llr(tracks, calib_gt)
        if results["gt_best_path"]:
            results["gt_best_path"]["rank_llr_gt"] = rank_of_path(
                scored_llr_gt, results["gt_best_path"]["tids"]
            )
            results["gt_best_path"]["score_llr_gt"] = next(
                (
                    p["score"]
                    for p in scored_llr_gt
                    if p["tids"] == results["gt_best_path"]["tids"]
                ),
                None,
            )

    # --- 可行性診斷 ---
    print("GT 可行性診斷…")
    diag = diagnose_gt_feasibility(tracks, gt["person_tids"])
    results["feasibility"] = diag
    diag_txt = out_dir / "gt_feasibility_20260507.txt"
    write_diagnose_txt(diag, diag_txt)
    print(f"可行性報告：{diag_txt}")
    print(
        f"結論：最多可涵蓋 {diag['max_gt_coverable']}/{diag['n_gt']} 條 GT"
    )

    out_json = out_dir / "evaluate_20260507.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"評估 JSON：{out_json}")

    # 簡短 Top-10 表
    for tag in ("old", "llr_prior", "llr_gt"):
        block = results.get(tag)
        if not block:
            continue
        print(f"\n===== {tag} Top-10 precision/recall =====")
        for r in block["top10"]:
            pp = r.get("path_probability")
            pp_s = f"P={pp:.4f}" if pp is not None else ""
            print(
                f"  #{r['rank']} prec={r['precision']:.2f} rec={r['recall']:.2f} "
                f"hit={r['n_hit']}/{r['n_path']} {pp_s}  {r['path']}"
            )

    return results


if __name__ == "__main__":
    main()
