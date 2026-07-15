# -*- coding: utf-8 -*-
"""
用人工 GT 校準 LLR 分布（僅分布，不改硬規則／候選池）
======================================================
輸入：query_filter_merge/{dataset}/ 的 26 條候選 track + GT JSON
輸出：
  ../output/path_enum_llr/calibration_gt0507.pkl
  ../output/path_enum_llr/calibration_gt0507_report.txt
  ../output/path_enum_llr/emb_same_diff_hist_gt0507.png

樣本：
  emb|same：GT×GT 跨鏡 cosine
  emb|diff：GT×非GT 全部配對
  dt|same：t_start 排序後相鄰且 edge_check 合法的轉移（按鏡頭對）
  dH|same：有 H 的 GT 重疊交接投影距離
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

DH_DIFF_UNIFORM_MAX = 800.0
# PRIOR-WEAK（2026-07-15）：原 0.5 → 1.0；tau 仍佔位。見 path_enum_llr.PRIOR_WEAK_NOTE
PRIOR_SIGMA = 1.0
MIN_SAMPLES_FIT = 20
PDF_FLOOR = 1e-12
SHRINK_K = 10
CALIB_SOURCE = "GT_20260507"


def _time_overlap(a: pes.Track, b: pes.Track) -> bool:
    return not (a.t_end < b.t_start or b.t_end < a.t_start)


def _fit_lognormal(samples: np.ndarray) -> dict | None:
    samples = samples[samples > 0]
    if len(samples) < 2:
        return None
    shape, loc, scale = stats.lognorm.fit(samples, floc=0)
    return {
        "family": "lognorm",
        "mu": float(np.log(scale)),
        "sigma": float(shape),
        "n": int(len(samples)),
    }


def _fit_normal(samples: np.ndarray) -> dict | None:
    if len(samples) < 2:
        return None
    mu, sigma = stats.norm.fit(samples)
    return {
        "family": "norm",
        "mu": float(mu),
        "sigma": max(float(sigma), 1e-4),
        "n": int(len(samples)),
    }


def _fit_halfnormal(samples: np.ndarray) -> dict | None:
    samples = samples[samples >= 0]
    if len(samples) < 2:
        return None
    _, scale = stats.halfnorm.fit(samples, floc=0)
    return {
        "family": "halfnorm",
        "sigma": max(float(scale), 1e-4),
        "n": int(len(samples)),
    }


def _tau_for_pair(cam_a: str, cam_b: str) -> float:
    hop = pes.hop_count(cam_a, cam_b)
    if hop is None:
        hop = 1
    return float(pes.tau(cam_a, cam_b, hop))


def _norm_fit_stats(arr: np.ndarray) -> dict:
    if len(arr) < 1:
        return {"n": 0, "mu": None, "sigma": None}
    if len(arr) < 2:
        return {"n": int(len(arr)), "mu": float(arr[0]), "sigma": None}
    mu, sigma = stats.norm.fit(arr)
    return {"n": int(len(arr)), "mu": float(mu), "sigma": float(max(sigma, 1e-4))}


def _emb_same_cross_cam(tracks: list) -> np.ndarray:
    vals: list[float] = []
    for i, u in enumerate(tracks):
        for v in tracks[i + 1 :]:
            if u.cam == v.cam:
                continue
            vals.append(pes.emb_sim(u, v))
    return np.asarray(vals, dtype=np.float64)


def collect_gt_samples(
    tracks: list,
    gt_tids: list[str],
    *,
    removed_mislabel: list[str] | None = None,
) -> dict:
    by_tid = {t.tid: t for t in tracks}
    gt_set = set(gt_tids)
    missing = [t for t in gt_tids if t not in by_tid]
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    non_gt = [t for t in tracks if t.tid not in gt_set]

    emb_same: list[float] = []
    emb_diff: list[float] = []
    dh_same: list[float] = []
    dt_same_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    time_overlaps = []

    # 剔除誤標前後 emb|same 對照（含 08_43 的舊 GT vs 現行 GT）
    removed = list(removed_mislabel or [])
    old_gt_tids = list(dict.fromkeys(list(gt_tids) + removed))
    old_gt_tracks = [by_tid[t] for t in old_gt_tids if t in by_tid]
    emb_same_before = _emb_same_cross_cam(old_gt_tracks)
    emb_same_after = _emb_same_cross_cam(gt_tracks)
    emb_same_compare = {
        "before_remove_08_43": _norm_fit_stats(emb_same_before),
        "after_remove_08_43": _norm_fit_stats(emb_same_after),
        "removed": removed,
    }

    # emb|same：GT×GT 跨鏡（已不含 08_43）
    emb_same = emb_same_after.tolist()

    # emb|diff：GT × 非GT（全部配對，含同鏡）→ 11×15
    for u in gt_tracks:
        for v in non_gt:
            emb_diff.append(pes.emb_sim(u, v))

    # sim|GT / sim|nonGT：對 query 的 track.sim（節點證據用）
    sim_gt = np.asarray([float(t.sim) for t in gt_tracks], dtype=np.float64)
    sim_nongt = np.asarray([float(t.sim) for t in non_gt], dtype=np.float64)

    # 時間重疊如實記錄；dH|same：有 H 的重疊交接
    for i, u in enumerate(gt_tracks):
        for v in gt_tracks[i + 1 :]:
            if not _time_overlap(u, v):
                continue
            ov = min(u.t_end, v.t_end) - max(u.t_start, v.t_start)
            time_overlaps.append({"a": u.tid, "b": v.tid, "overlap_sec": ov})
            ok_h, d = pes.same_object_h(u, v)
            if ok_h and d is not None:
                dh_same.append(float(d))

    # dt|same：t_start 排序相鄰且 edge_check 合法
    ordered = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))
    dt_edges = []
    for u, v in zip(ordered, ordered[1:]):
        ok, reason, dt, hop, emb, h_dist = pes.edge_check(u, v)
        if not ok:
            continue
        key = tuple(sorted((u.cam, v.cam)))
        dt_same_by_pair[key].append(float(dt))
        dt_edges.append(
            {
                "from": u.tid,
                "to": v.tid,
                "pair": f"{key[0]}|{key[1]}",
                "dt": float(dt),
                "hop": hop,
                "emb": float(emb),
            }
        )

    return {
        "emb_same": np.asarray(emb_same, dtype=np.float64),
        "emb_diff": np.asarray(emb_diff, dtype=np.float64),
        "dh_same": np.asarray(dh_same, dtype=np.float64),
        "sim_gt": sim_gt,
        "sim_nongt": sim_nongt,
        "dt_same_by_pair": {k: np.asarray(v, dtype=np.float64) for k, v in dt_same_by_pair.items()},
        "emb_gate_for_dt": None,
        "missing_gt": missing,
        "n_gt": len(gt_tracks),
        "n_non_gt": len(non_gt),
        "time_overlaps": time_overlaps,
        "dt_edges": dt_edges,
        "emb_same_compare": emb_same_compare,
        "counts": {
            "n_emb_same": len(emb_same),
            "n_emb_diff": len(emb_diff),
            "n_dh_same": len(dh_same),
            "n_sim_gt": int(len(sim_gt)),
            "n_sim_nongt": int(len(sim_nongt)),
            "n_dt_pairs_total": int(sum(len(v) for v in dt_same_by_pair.values())),
            "n_gt": len(gt_tracks),
            "n_non_gt": len(non_gt),
            "n_time_overlaps": len(time_overlaps),
        },
    }


def fit_calibration(samples: dict) -> dict:
    n_emb_same = int(len(samples["emb_same"]))
    n_emb_diff = int(len(samples["emb_diff"]))
    n_dh_same = int(len(samples["dh_same"]))

    emb_same = _fit_normal(samples["emb_same"])
    emb_diff = _fit_normal(samples["emb_diff"])
    if emb_same is None:
        emb_same = {
            "family": "norm",
            "mu": 0.95,
            "sigma": 0.03,
            "n": n_emb_same,
            "prior": True,
            "reason": "n<2 cannot fit Normal",
        }
    if emb_diff is None:
        emb_diff = {
            "family": "norm",
            "mu": 0.70,
            "sigma": 0.10,
            "n": n_emb_diff,
            "prior": True,
            "reason": "n<2 cannot fit Normal",
        }

    dh_same = _fit_halfnormal(samples["dh_same"])
    if dh_same is None:
        dh_same = {
            "family": "halfnorm",
            "sigma": 40.0,
            "n": n_dh_same,
            "prior": True,
            "reason": "n<2 cannot fit HalfNormal",
        }

    # 收縮權重如實寫入（供報告；LLR 執行時用 n 重算 w=n/(n+10)）
    for dist in (emb_same, emb_diff, dh_same):
        n = int(dist.get("n") or 0)
        dist["shrink_w"] = float(n) / float(n + SHRINK_K)

    sim_gt = _fit_normal(samples["sim_gt"])
    sim_nongt = _fit_normal(samples["sim_nongt"])
    if sim_gt is None:
        sim_gt = {
            "family": "norm",
            "mu": float(np.mean(samples["sim_gt"])) if len(samples["sim_gt"]) else 0.9,
            "sigma": 0.05,
            "n": int(len(samples["sim_gt"])),
            "prior": len(samples["sim_gt"]) < 2,
        }
    if sim_nongt is None:
        sim_nongt = {
            "family": "norm",
            "mu": float(np.mean(samples["sim_nongt"])) if len(samples["sim_nongt"]) else 0.8,
            "sigma": 0.05,
            "n": int(len(samples["sim_nongt"])),
            "prior": len(samples["sim_nongt"]) < 2,
        }
    for dist in (sim_gt, sim_nongt):
        n = int(dist.get("n") or 0)
        dist["shrink_w"] = float(n) / float(n + SHRINK_K)

    dt_by_pair = {}
    prior_pairs = []
    for key in sorted(pes.ADJACENT):
        if key in pes.OVERLAP_PAIRS:
            continue
        arr = samples["dt_same_by_pair"].get(key, np.asarray([], dtype=np.float64))
        if len(arr) >= MIN_SAMPLES_FIT:
            fit = _fit_lognormal(arr)
            if fit is not None:
                fit["shrink_w"] = 1.0
                dt_by_pair[key] = fit
                continue
        tau0 = _tau_for_pair(key[0], key[1])
        dt_by_pair[key] = {
            "family": "lognorm",
            "mu": float(np.log(max(tau0, 1e-3))),
            "sigma": PRIOR_SIGMA,
            "n": int(len(arr)),
            "prior": True,
            "prior_physical": True,
            "prior_weak": True,
            "tau": tau0,
            "shrink_w": 1.0,
            "note": "PRIOR-WEAK",
        }
        prior_pairs.append(
            {
                "pair": key,
                "n_samples": int(len(arr)),
                "tau": tau0,
                "sigma": PRIOR_SIGMA,
                "note": "PRIOR-WEAK",
            }
        )

    for key, arr in samples["dt_same_by_pair"].items():
        if key in dt_by_pair:
            continue
        if len(arr) >= MIN_SAMPLES_FIT:
            fit = _fit_lognormal(arr)
            if fit is not None:
                fit["shrink_w"] = 1.0
                dt_by_pair[key] = fit
                continue
        # hop2 等非 ADJACENT 鏡頭對：樣本不足仍記先驗 + 實際 n
        tau0 = _tau_for_pair(key[0], key[1])
        dt_by_pair[key] = {
            "family": "lognorm",
            "mu": float(np.log(max(tau0, 1e-3))),
            "sigma": PRIOR_SIGMA,
            "n": int(len(arr)),
            "prior": True,
            "prior_physical": True,
            "prior_weak": True,
            "tau": tau0,
            "shrink_w": 1.0,
            "note": "PRIOR-WEAK",
        }
        prior_pairs.append(
            {
                "pair": key,
                "n_samples": int(len(arr)),
                "tau": tau0,
                "sigma": PRIOR_SIGMA,
                "note": "PRIOR-WEAK",
            }
        )

    return {
        "emb_same": emb_same,
        "emb_diff": emb_diff,
        "dh_same": dh_same,
        "dh_diff": {"family": "uniform", "low": 0.0, "high": DH_DIFF_UNIFORM_MAX, "n": None},
        "dt_diff": {"family": "uniform", "low": 0.0, "high": float(pes.DT_MAX), "n": None},
        "sim_gt": sim_gt,
        "sim_nongt": sim_nongt,
        "dt_same_by_pair": {f"{a}|{b}": v for (a, b), v in dt_by_pair.items()},
        "prior_pairs": prior_pairs,
        "meta": {
            "CALIB_SOURCE": CALIB_SOURCE,
            "min_samples_fit": MIN_SAMPLES_FIT,
            "prior_sigma": PRIOR_SIGMA,
            "prior_weak_note": "transit dt prior LogNormal sigma=1.0 (was 0.5); tau placeholder",
            "pdf_floor": PDF_FLOOR,
            "shrink_k": SHRINK_K,
            "counts": samples["counts"],
            "missing_gt": samples["missing_gt"],
            "time_overlaps": samples["time_overlaps"],
            "dt_edges": samples["dt_edges"],
            "emb_same_compare": samples["emb_same_compare"],
            "warning": "IN-SAMPLE：校準與評估同一資料集 0507，僅供診斷",
        },
    }


def compute_transition_prior(tracks: list, gt_tids: list[str]) -> dict:
    """
    p_edge = GT真轉移邊數 / 全部合法邊數。
    GT真轉移 = 純 GT 超節點按 t_start 排序後相鄰且 _best_member_edge 合法。
    全部合法邊 = 全候選超節點圖上合法邊數（同聯集 dt 語意）。
    """
    import itertools

    import llr_gate_config as gates
    import path_enum_llr as llr

    gates.apply_llr_emb_gates(True)
    supers, srep = llr.build_supernodes(tracks)
    n_legal = 0
    for i, j in itertools.permutations(range(len(supers)), 2):
        best, _ = llr._best_member_edge(supers[i], supers[j])
        if best is not None:
            n_legal += 1

    gt_set = set(gt_tids)
    gt_supers = [
        s
        for s in supers
        if set(s.tids).issubset(gt_set) and len(s.tids) > 0
    ]
    gt_supers.sort(key=lambda s: (s.t_start, s.t_end, s.sid))
    gt_trans = []
    for u, v in zip(gt_supers, gt_supers[1:]):
        best, _ = llr._best_member_edge(u, v)
        if best is not None:
            uu, vv, dt, hop, emb, h_dist = best
            gt_trans.append(
                {
                    "from": u.label,
                    "to": v.label,
                    "via": f"{uu.tid}->{vv.tid}",
                    "dt": float(dt),
                    "hop": hop,
                    "emb": float(emb),
                }
            )
    n_gt = len(gt_trans)
    p_edge = float(n_gt) / float(n_legal) if n_legal > 0 else 0.0
    return {
        "p_edge": p_edge,
        "n_gt_true_transitions": n_gt,
        "n_legal_edges": n_legal,
        "formula": "p_edge = n_gt_true_transitions / n_legal_edges",
        "gt_transitions": gt_trans,
        "n_gt_supers": len(gt_supers),
        "multi_only": srep.get("multi_only"),
        "ln_p_edge": float(math.log(p_edge)) if p_edge > 0 else None,
        "note": (
            "GT真轉移=純GT超節點時間相鄰且合法；"
            "合法邊=全圖超節點邊（聯集dt）。2026-07-15"
        ),
    }


def save_emb_histogram(emb_same: np.ndarray, emb_diff: np.ndarray, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 41)
    if len(emb_same):
        ax.hist(
            emb_same,
            bins=bins,
            alpha=0.55,
            label=f"same GT×GT cross-cam (n={len(emb_same)})",
            color="tab:green",
            density=True,
        )
    if len(emb_diff):
        ax.hist(
            emb_diff,
            bins=bins,
            alpha=0.55,
            label=f"diff GT×nonGT (n={len(emb_diff)})",
            color="tab:red",
            density=True,
        )
    ax.set_xlabel("embedding cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"emb | same vs diff  [{CALIB_SOURCE}]")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_report(calib: dict, samples: dict, hist_png: Path, out_txt: Path) -> None:
    lines = []
    lines.append("=== calibration from GT ===")
    lines.append(f"CALIB_SOURCE={CALIB_SOURCE}")
    lines.append("WARNING: IN-SAMPLE（校準與評估同一資料集），結論僅供診斷")
    lines.append(f"counts: {calib['meta']['counts']}")
    lines.append(f"missing_gt: {calib['meta']['missing_gt']}")
    lines.append("")
    cmp_ = calib["meta"].get("emb_same_compare") or {}
    lines.append("--- emb|same (μ,σ) 剔除 K8-08_43 前後對照 ---")
    b = cmp_.get("before_remove_08_43") or {}
    a = cmp_.get("after_remove_08_43") or {}
    lines.append(
        f"  before (含 08_43): n={b.get('n')} mu={b.get('mu')} sigma={b.get('sigma')}"
    )
    lines.append(
        f"  after  (剔除後):   n={a.get('n')} mu={a.get('mu')} sigma={a.get('sigma')}"
    )
    lines.append("")
    lines.append("--- emb|same (Normal) ---")
    lines.append(str(calib["emb_same"]))
    lines.append("--- emb|diff (Normal) ---")
    lines.append(str(calib["emb_diff"]))
    lines.append("--- sim|GT / sim|nonGT (query sim, Normal) ---")
    lines.append(str(calib.get("sim_gt")))
    lines.append(str(calib.get("sim_nongt")))
    lines.append("--- dH|same (HalfNormal) ---")
    lines.append(str(calib["dh_same"]))
    lines.append("--- dH|diff / dt|diff (fixed Uniform) ---")
    lines.append(str(calib["dh_diff"]))
    lines.append(str(calib["dt_diff"]))
    lines.append("")
    lines.append("--- dt|same by camera pair ---")
    for k, v in sorted(calib["dt_same_by_pair"].items()):
        tag = " PRIOR" if v.get("prior") else ""
        lines.append(
            f"  {k}: n={v.get('n', 0)} mu={v.get('mu'):.4f} "
            f"sigma={v.get('sigma'):.4f} w={v.get('shrink_w', float('nan')):.3f}{tag}"
        )
    lines.append("")
    lines.append(f"prior pairs: {len(calib['prior_pairs'])}")
    for p in calib["prior_pairs"]:
        lines.append(f"  {p['pair']}: n={p['n_samples']} tau={p['tau']:.2f}")
    lines.append("")
    tp = calib.get("transition_prior") or {}
    lines.append("--- transition prior p_edge ---")
    lines.append(f"  formula: {tp.get('formula')}")
    lines.append(
        f"  n_gt_true_transitions={tp.get('n_gt_true_transitions')}  "
        f"n_legal_edges={tp.get('n_legal_edges')}  "
        f"p_edge={tp.get('p_edge')}  ln(p)={tp.get('ln_p_edge')}"
    )
    for e in tp.get("gt_transitions") or []:
        lines.append(
            f"  GT transfer {e['from']} -> {e['to']} via {e['via']}  "
            f"dt={e['dt']:.2f} hop={e['hop']} emb={e['emb']:.3f}"
        )
    lines.append("")
    lines.append(f"GT time overlaps reported: {len(samples['time_overlaps'])}")
    for o in samples["time_overlaps"]:
        lines.append(f"  {o['a']} ↔ {o['b']}  overlap={o['overlap_sec']:.2f}s")
    lines.append("")
    lines.append(f"histogram: {hist_png}")
    lines.append(f"shrink_k={SHRINK_K}  (w=n/(n+K) recorded per distribution)")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="Calibrate LLR distributions from human GT")
    p.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "ground_truth_20260507.json",
    )
    p.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr",
    )
    args = p.parse_args(argv)

    merge_dir = args.merge_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    pes.SIM_MIN = float(args.sim_min)
    pes.configure_for_input(str(merge_dir))
    tracks = pes.load_tracks(str(merge_dir))
    print(f"載入 {len(tracks)} 條 track；GT={len(gt['person_tids'])}")

    samples = collect_gt_samples(
        tracks,
        gt["person_tids"],
        removed_mislabel=list(gt.get("removed_mislabel") or ["K8-08_43"]),
    )
    print(
        f"emb_same={samples['counts']['n_emb_same']}  "
        f"emb_diff={samples['counts']['n_emb_diff']}  "
        f"dh_same={samples['counts']['n_dh_same']}  "
        f"sim_gt={samples['counts']['n_sim_gt']}  "
        f"sim_nongt={samples['counts']['n_sim_nongt']}  "
        f"dt={samples['counts']['n_dt_pairs_total']}  "
        f"overlaps={samples['counts']['n_time_overlaps']}"
    )
    cmp_ = samples["emb_same_compare"]
    print(
        "emb|same before/after 08_43:",
        cmp_["before_remove_08_43"],
        "->",
        cmp_["after_remove_08_43"],
    )
    if samples["missing_gt"]:
        print("缺失 GT track：", samples["missing_gt"])

    calib = fit_calibration(samples)
    calib["dataset"] = merge_dir.name
    calib["input_dir"] = str(merge_dir)
    calib["gt_path"] = str(args.gt.resolve())

    # 轉移先驗（需在 emb 門檻覆寫後算；用聯集 dt 超節點邊）
    tp = compute_transition_prior(tracks, gt["person_tids"])
    calib["transition_prior"] = tp
    print(
        f"transition prior: p_edge={tp['p_edge']:.6f} = "
        f"{tp['n_gt_true_transitions']}/{tp['n_legal_edges']}  "
        f"ln(p)={tp.get('ln_p_edge')}"
    )

    pkl_path = out_dir / "calibration_gt0507.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(calib, f)

    hist_png = out_dir / "emb_same_diff_hist_gt0507.png"
    save_emb_histogram(samples["emb_same"], samples["emb_diff"], hist_png)

    report = out_dir / "calibration_gt0507_report.txt"
    write_report(calib, samples, hist_png, report)

    print(f"寫入：{pkl_path}")
    print(f"寫入：{report}")
    print(f"寫入：{hist_png}")
    print(f"CALIB_SOURCE={CALIB_SOURCE}")


if __name__ == "__main__":
    main()
