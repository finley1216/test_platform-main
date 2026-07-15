# -*- coding: utf-8 -*-
"""
0528 資料集內自校準（不碰 GT）
==============================
正樣本 = 超節點成員對（coexistence_merge：OVERLAP 或 H×dH<80 確認）
負樣本 =
  - 同鏡同時段對（時間重疊）
  - 跨鏡、時間共存、但鏡頭對既非 OVERLAP 亦非 ADJACENT

擬合 emb|same、emb|diff（及正樣本 dH|same 若有）。
其餘分布（dt、sim）自 0507 校準繼承，標注來自 GT 校準檔；
本腳本不讀取 / 不依賴 0528 GT。

輸出：
  calibration_self0528.pkl
  calibration_self0528_report.txt
  emb_same_diff_hist_self0528.png
"""

from __future__ import annotations

import argparse
import itertools
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import llr_gate_config as gates  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

SHRINK_K = 10
DH_DIFF_UNIFORM_MAX = 800.0
CALIB_SOURCE = "SELF_20260528_NO_GT"


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
    if len(samples) < 2:
        return None
    sigma = float(stats.halfnorm.fit(samples, floc=0)[1])
    return {
        "family": "halfnorm",
        "sigma": max(sigma, 1e-4),
        "n": int(len(samples)),
    }


def _time_overlap_sec(a: pes.Track, b: pes.Track) -> float:
    return max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))


def _pair_key(cam_a: str, cam_b: str) -> tuple[str, str]:
    return tuple(sorted((cam_a, cam_b)))


def collect_self_samples(tracks: list) -> dict:
    """正／負樣本皆不使用 GT。"""
    gates.apply_llr_emb_gates(True)
    supers, srep = llr.build_supernodes(tracks)

    emb_same = []
    dh_same = []
    pos_pairs = []
    for sn in supers:
        if len(sn.members) < 2:
            continue
        for u, v in itertools.combinations(sn.members, 2):
            ok, reason = llr.coexistence_merge(u, v)
            if not ok:
                continue
            e = float(pes.emb_sim(u, v))
            emb_same.append(e)
            ok_h, d = pes.same_object_h(u, v)
            dh = float(d) if (ok_h and d is not None) else None
            if dh is not None:
                dh_same.append(dh)
            pos_pairs.append(
                {
                    "a": u.tid,
                    "b": v.tid,
                    "emb": e,
                    "dH": dh,
                    "reason": reason,
                    "super": sn.label,
                }
            )

    emb_diff = []
    neg_pairs = []
    for u, v in itertools.combinations(tracks, 2):
        ov = _time_overlap_sec(u, v)
        if ov <= 0.0:
            # 單幀點重疊：若一端長度≈0，允許可點命中
            tok, _, _ = llr._coexistence_time_ok(u, v)
            if not tok:
                continue
            ov = 1e-6

        if u.cam == v.cam:
            e = float(pes.emb_sim(u, v))
            emb_diff.append(e)
            neg_pairs.append(
                {
                    "a": u.tid,
                    "b": v.tid,
                    "kind": "same_cam_overlap",
                    "emb": e,
                    "overlap_sec": ov,
                }
            )
            continue

        key = _pair_key(u.cam, v.cam)
        if key in pes.OVERLAP_PAIRS or key in pes.ADJACENT:
            continue
        # 跨鏡共存且拓撲上非相鄰／非重疊 → 硬性「應為不同人」
        e = float(pes.emb_sim(u, v))
        emb_diff.append(e)
        neg_pairs.append(
            {
                "a": u.tid,
                "b": v.tid,
                "kind": "cross_cam_non_adj_non_ov_coexist",
                "cams": f"{u.cam}|{v.cam}",
                "emb": e,
                "overlap_sec": ov,
            }
        )

    return {
        "emb_same": np.asarray(emb_same, dtype=np.float64),
        "emb_diff": np.asarray(emb_diff, dtype=np.float64),
        "dh_same": np.asarray(dh_same, dtype=np.float64),
        "pos_pairs": pos_pairs,
        "neg_pairs": neg_pairs,
        "super_report": {
            "n_supernodes": srep["n_supernodes"],
            "n_merged_pairs": srep["n_merged_pairs"],
            "multi_only": srep.get("multi_only"),
        },
        "counts": {
            "n_emb_same": len(emb_same),
            "n_emb_diff": len(emb_diff),
            "n_dh_same": len(dh_same),
            "n_pos_pairs": len(pos_pairs),
            "n_neg_same_cam": sum(1 for p in neg_pairs if p["kind"] == "same_cam_overlap"),
            "n_neg_cross": sum(
                1 for p in neg_pairs if p["kind"] == "cross_cam_non_adj_non_ov_coexist"
            ),
            "n_tracks": len(tracks),
        },
    }


def compute_self_transition_prior(tracks: list) -> dict:
    """
    不碰 GT：以多成員超節點（幾何確認）為錨，
    p_edge = n_anchor_true_transitions / n_legal_edges。
    """
    gates.apply_llr_emb_gates(True)
    supers, srep = llr.build_supernodes(tracks)
    n_legal = 0
    for i, j in itertools.permutations(range(len(supers)), 2):
        best, _ = llr._best_member_edge(supers[i], supers[j])
        if best is not None:
            n_legal += 1

    anchors = [s for s in supers if len(s.members) > 1]
    anchors.sort(key=lambda s: (s.t_start, s.t_end, s.sid))
    trans = []
    for u, v in zip(anchors, anchors[1:]):
        best, _ = llr._best_member_edge(u, v)
        if best is not None:
            uu, vv, dt, hop, emb, h_dist = best
            trans.append(
                {
                    "from": u.label,
                    "to": v.label,
                    "via": f"{uu.tid}->{vv.tid}",
                    "dt": float(dt),
                    "hop": hop,
                    "emb": float(emb),
                }
            )
    n_true = len(trans)
    p_edge = float(n_true) / float(n_legal) if n_legal > 0 else 0.0
    return {
        "p_edge": p_edge,
        "n_self_true_transitions": n_true,
        "n_gt_true_transitions": n_true,  # 相容欄位名供 edge_llr 讀取注記
        "n_legal_edges": n_legal,
        "formula": "p_edge = n_multi_super_adjacent_legal / n_legal_edges (NO GT)",
        "self_transitions": trans,
        "n_anchor_supers": len(anchors),
        "multi_only": srep.get("multi_only"),
        "ln_p_edge": float(math.log(p_edge)) if p_edge > 0 else None,
        "note": (
            "自校準先驗：多成員超節點時間相鄰且合法 / 全圖合法邊；"
            "未使用 GT。2026-07-15"
        ),
    }


def fit_self_calibration(samples: dict, base_calib: dict) -> dict:
    emb_same = _fit_normal(samples["emb_same"])
    emb_diff = _fit_normal(samples["emb_diff"])
    if emb_same is None:
        emb_same = {
            "family": "norm",
            "mu": 0.95,
            "sigma": 0.03,
            "n": int(len(samples["emb_same"])),
            "prior": True,
            "reason": "n<2",
        }
    if emb_diff is None:
        emb_diff = {
            "family": "norm",
            "mu": 0.70,
            "sigma": 0.10,
            "n": int(len(samples["emb_diff"])),
            "prior": True,
            "reason": "n<2",
        }
    for d in (emb_same, emb_diff):
        n = int(d.get("n") or 0)
        d["shrink_w"] = float(n) / float(n + SHRINK_K)

    dh_same = _fit_halfnormal(samples["dh_same"])
    if dh_same is None:
        # 繼承 0507（樣本不足）
        dh_same = dict(base_calib.get("dh_same") or {
            "family": "halfnorm",
            "sigma": 40.0,
            "n": 0,
            "prior": True,
        })
        dh_same["inherited_from"] = "calibration_gt0507"
    else:
        dh_same["shrink_w"] = float(dh_same["n"]) / float(dh_same["n"] + SHRINK_K)

    # sim_*：自校準不做節點證據擬合；執行時可選併入 0507 版
    out = {
        "emb_same": emb_same,
        "emb_diff": emb_diff,
        "dh_same": dh_same,
        "dh_diff": dict(base_calib.get("dh_diff") or {
            "family": "uniform", "low": 0.0, "high": DH_DIFF_UNIFORM_MAX, "n": None
        }),
        "dt_diff": dict(base_calib.get("dt_diff") or {
            "family": "uniform", "low": 0.0, "high": float(pes.DT_MAX), "n": None
        }),
        "dt_same_by_pair": dict(base_calib.get("dt_same_by_pair") or {}),
        "prior_pairs": list(base_calib.get("prior_pairs") or []),
        # 故意不放 sim_gt / sim_nongt（預設 node evidence off；
        # 實驗可再 merge 0507 的 sim_*）
        "meta": {
            "CALIB_SOURCE": CALIB_SOURCE,
            "shrink_k": SHRINK_K,
            "counts": samples["counts"],
            "pos_pairs": samples["pos_pairs"],
            "neg_pairs_sample": samples["neg_pairs"][:50],
            "n_neg_pairs_total": len(samples["neg_pairs"]),
            "super_report": samples["super_report"],
            "inherited": {
                "dt_same_by_pair": "calibration_gt0507 (dt_scoring=off 時不生效)",
                "dh_diff": "calibration_gt0507",
                "dt_diff": "calibration_gt0507",
                "sim_gt_sim_nongt": "absent_by_default; merge for node_ev=0507",
            },
            "warning": "SELF-CALIB 0528：正負樣本不使用 GT；僅供診斷",
            "compare_to_gt0507": {
                "emb_same": base_calib.get("emb_same"),
                "emb_diff": base_calib.get("emb_diff"),
            },
        },
    }
    return out


def save_hist(emb_same, emb_diff, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 41)
    if len(emb_same):
        ax.hist(emb_same, bins=bins, alpha=0.55, label=f"same supernode (n={len(emb_same)})")
    if len(emb_diff):
        ax.hist(emb_diff, bins=bins, alpha=0.55, label=f"diff coexist (n={len(emb_diff)})")
    ax.set_xlabel("embedding cosine")
    ax.set_ylabel("count")
    ax.set_title("0528 self-calib emb|same vs emb|diff (NO GT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_report(
    calib: dict,
    samples: dict,
    prior: dict,
    base_calib: dict,
    out_txt: Path,
) -> str:
    es, ed = calib["emb_same"], calib["emb_diff"]
    b_es, b_ed = base_calib["emb_same"], base_calib["emb_diff"]
    lines = [
        "=== calibration_self0528 ===",
        f"source: {CALIB_SOURCE}",
        "GT: NOT USED",
        "",
        "## samples",
        f"  emb|same (supernode members): n={es['n']}  mu={es['mu']:.4f}  sigma={es['sigma']:.4f}  w={es['shrink_w']:.4f}",
        f"  emb|diff (same-cam + non-adj/ov coexist): n={ed['n']}  mu={ed['mu']:.4f}  sigma={ed['sigma']:.4f}  w={ed.get('shrink_w', float('nan')):.4f}",
        f"  dH|same: n={samples['counts']['n_dh_same']}",
        f"  neg breakdown: same_cam={samples['counts']['n_neg_same_cam']}  cross={samples['counts']['n_neg_cross']}",
        f"  multi supers: {samples['super_report'].get('multi_only')}",
        "",
        "## vs calibration_gt0507",
        f"  emb|same 0507: n={b_es['n']} mu={b_es['mu']:.4f} sigma={b_es['sigma']:.4f} w={b_es['shrink_w']:.4f}",
        f"  emb|diff 0507: n={b_ed['n']} mu={b_ed['mu']:.4f} sigma={b_ed['sigma']:.4f} w={b_ed['shrink_w']:.4f}",
        f"  Δmu_same={es['mu']-b_es['mu']:+.4f}  Δmu_diff={ed['mu']-b_ed['mu']:+.4f}",
        f"  Δw_emb (same.n): {es['shrink_w']-b_es['shrink_w']:+.4f}",
        "",
        "## self transition prior (NO GT)",
        f"  p_edge={prior['p_edge']:.6f}  ln={prior.get('ln_p_edge')}",
        f"  n_true={prior['n_self_true_transitions']}  n_legal={prior['n_legal_edges']}",
        f"  formula: {prior['formula']}",
        "",
        "## positive pairs",
    ]
    for p in samples["pos_pairs"]:
        lines.append(
            f"  {p['a']}↔{p['b']} emb={p['emb']:.4f} dH={p['dH']} ({p['reason']})"
        )
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
    )
    ap.add_argument(
        "--base-calib",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "calibration_gt0507.pkl",
        help="繼承 dt/dh_diff 結構用；不讀其 emb 當作自校準結果",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr",
    )
    ap.add_argument("--sim-min", type=float, default=0.85)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pes.SIM_MIN = float(args.sim_min)
    pes.configure_for_input(str(args.merge_dir))
    tracks = pes.load_tracks(str(args.merge_dir))
    print(f"載入 tracks={len(tracks)}  SIM_MIN={pes.SIM_MIN}")
    base = pickle.loads(Path(args.base_calib).read_bytes())

    samples = collect_self_samples(tracks)
    calib = fit_self_calibration(samples, base)
    prior = compute_self_transition_prior(tracks)
    calib["transition_prior"] = prior

    pkl = out_dir / "calibration_self0528.pkl"
    pkl.write_bytes(pickle.dumps(calib))
    report = write_report(
        calib, samples, prior, base, out_dir / "calibration_self0528_report.txt"
    )
    save_hist(
        samples["emb_same"],
        samples["emb_diff"],
        out_dir / "emb_same_diff_hist_self0528.png",
    )
    print(report)
    print(f"寫入 {pkl}")


if __name__ == "__main__":
    main()
