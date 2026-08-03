#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
車輛 0507：GT 校準 → calibration_vehicle_gt0507.pkl；M0 凍結設定首跑；報告。
鐵則：不改 track_path.py；GT 僅校準／評估；不調參。
"""
from __future__ import annotations

import json
import math
import pickle
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKING = _REPO_ROOT / "scripts" / "tracking"
for _p in (_REPO_ROOT, _TRACKING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from PIL import Image, ImageDraw

import track_path_m0 as tp

OUT_ROOT = tp.OUTPUT_ROOT / "v1.0"
MERGE_DIR = tp.QUERY_FILTER_OUTPUT_ROOT / "車輛追蹤_20260507"
GT_PATH = OUT_ROOT / "ground_truth_vehicle_20260507.json"
PERSON_CALIB = OUT_ROOT / "calibration_gt0507.pkl"
RUN_DIR = OUT_ROOT / "vehicle_m0_0507"
SIM_MIN = 0.85  # 與人員 M0 / 標註池一致；凍結設定


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mu": None, "sigma": None}
    if len(arr) == 1:
        return {"n": 1, "mu": float(arr[0]), "sigma": None}
    mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=0))
    # 與 fit Normal 對齊：scipy norm.fit 用 MLE（σ 同 ddof=0）
    return {"n": int(len(arr)), "mu": mu, "sigma": max(sigma, 1e-4)}


def list_same_cam_excluded(gt_tracks: list) -> list[dict]:
    """emb_same 排除的同鏡配對。"""
    excluded = []
    for i, u in enumerate(gt_tracks):
        for v in gt_tracks[i + 1 :]:
            if u.cam == v.cam:
                excluded.append(
                    {
                        "a": u.tid,
                        "b": v.tid,
                        "cam": u.cam,
                        "emb": float(tp.emb_sim(u, v)),
                    }
                )
    return excluded


def gt_true_transition_dts(gt_tracks: list) -> list[dict]:
    """
    車輛 GT 真轉移 dt：按 t_start 排序後相鄰對，
    dt = max(v.t_start - u.t_end, 0)（與 edge_check 語意一致），
    不論是否通過 DT_MAX／拓撲／emb。
    """
    ordered = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))
    rows = []
    for u, v in zip(ordered, ordered[1:]):
        dt_raw = v.t_start - u.t_end
        dt = max(dt_raw, 0.0)
        hop = tp.hop_count(u.cam, v.cam)
        ok, reason, dt_e, hop_e, emb, h_dist = tp.edge_check(u, v)
        rows.append(
            {
                "from": u.tid,
                "to": v.tid,
                "from_cam": u.cam,
                "to_cam": v.cam,
                "dt_raw": float(dt_raw),
                "dt": float(dt),
                "hop": hop,
                "edge_ok": bool(ok),
                "edge_reason": reason or "",
                "emb": float(emb) if emb is not None else None,
                "exceeds_dt_max": bool(dt > tp.DT_MAX),
                "u_span": [u.t_start, u.t_end],
                "v_span": [v.t_start, v.t_end],
            }
        )
    return rows


def save_compare_hist(
    veh_same, veh_diff, per_same, per_diff, out_png: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, same, diff, title in (
        (axes[0], veh_same, veh_diff, "vehicle 0507"),
        (axes[1], per_same, per_diff, "person 0507"),
    ):
        if len(same):
            ax.hist(
                same,
                bins=bins,
                alpha=0.55,
                density=True,
                color="tab:green",
                label=f"emb_same n={len(same)}",
            )
        if len(diff):
            ax.hist(
                diff,
                bins=bins,
                alpha=0.55,
                density=True,
                color="tab:red",
                label=f"emb_diff n={len(diff)}",
            )
        ax.set_title(title)
        ax.set_xlabel("embedding similarity")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("density")
    fig.suptitle("emb_same / emb_diff: vehicle vs person")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def render_gt_colored_collage(
    merge_dir: Path,
    top: dict,
    gt_set: set[str],
    out_png: Path,
    *,
    title_prefix: str = "vehicle M0 Top-1",
) -> Path:
    tids = list(top.get("tids") or [])
    edges = list(top.get("edges") or [])
    n = len(tids)
    cell_w, cell_h, arrow_w = 150, 210, 54
    tw, th = 134, 100
    margin, title_h, foot_h, legend_h = 16, 52, 48, 28
    width = margin * 2 + max(n, 1) * cell_w + max(0, n - 1) * arrow_w
    height = title_h + legend_h + cell_h + foot_h + margin
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_t = tp._font(16)
    font_s = tp._font(12)
    font_xs = tp._font(10)

    prec = sum(1 for t in tids if t in gt_set) / n if n else 0.0
    rec = sum(1 for t in tids if t in gt_set) / max(len(gt_set), 1)
    p = float(top.get("path_probability") or 0.0)
    title = (
        f"{title_prefix}  score={float(top.get('score') or 0):.2f}  "
        f"P={p:.4f}  prec={prec:.3f} rec={rec:.3f}  "
        f"(SIM>={tp.SIM_MIN})  [{merge_dir.name}]"
    )
    draw.text((margin, 8), title, fill=(20, 20, 20), font=font_t)
    ly = title_h - 4
    draw.rectangle([margin, ly, margin + 14, ly + 12], outline=(34, 139, 34), width=2)
    draw.text((margin + 18, ly - 1), "GT", fill=(34, 139, 34), font=font_xs)
    draw.rectangle(
        [margin + 55, ly, margin + 69, ly + 12], outline=(200, 40, 40), width=2
    )
    draw.text((margin + 73, ly - 1), "non-GT", fill=(200, 40, 40), font=font_xs)

    y0 = title_h + legend_h
    for i, tid in enumerate(tids):
        cam, tid_s = tid.rsplit("_", 1)
        tid_i = int(tid_s)
        try:
            tr, crops = tp._crop_paths_for_track(merge_dir, cam, tid_i)
        except Exception:
            tr, crops = {}, []
        rep = tp._pick_rep_crop(crops) if crops else None
        x = margin + i * (cell_w + arrow_w)
        border = (34, 139, 34) if tid in gt_set else (200, 40, 40)
        draw.rectangle(
            [x, y0, x + cell_w - 1, y0 + cell_h - 1], outline=border, width=3
        )
        draw.text((x + 6, y0 + 4), tid, fill=border, font=font_s)
        if rep is not None:
            timg = tp._thumb(rep, (tw, th))
            img.paste(timg, (x + (cell_w - tw) // 2, y0 + 26))
        else:
            draw.text((x + 16, y0 + 90), "(no crop)", fill=(160, 0, 0), font=font_s)
        sim = float(tr.get("similarity", 0.0)) if tr else 0.0
        draw.text(
            (x + 6, y0 + cell_h - 40), f"sim={sim:.3f}", fill=(30, 30, 30), font=font_xs
        )
        tag = "GT" if tid in gt_set else "nonGT"
        draw.text((x + 6, y0 + cell_h - 22), tag, fill=border, font=font_xs)

        if i < len(edges):
            e = edges[i]
            ax0 = x + cell_w
            ax1 = ax0 + arrow_w
            mid_y = y0 + cell_h // 2
            draw.line([(ax0 + 6, mid_y), (ax1 - 10, mid_y)], fill=(0, 0, 0), width=2)
            draw.polygon(
                [(ax1 - 10, mid_y - 6), (ax1 - 2, mid_y), (ax1 - 10, mid_y + 6)],
                fill=(0, 0, 0),
            )
            sc = float(e.get("score") or 0.0)
            sc_color = (0, 128, 0) if sc >= 0 else (180, 0, 0)
            draw.text(
                (ax0 + 2, mid_y - 38),
                f"hop={e.get('hop')}",
                fill=(40, 40, 40),
                font=font_xs,
            )
            draw.text(
                (ax0 + 2, mid_y - 22),
                f"dt={float(e.get('dt') or 0):.1f}s",
                fill=(40, 40, 40),
                font=font_xs,
            )
            draw.text(
                (ax0 + 2, mid_y + 8),
                f"emb={float(e.get('emb') or 0):.3f}",
                fill=(40, 40, 40),
                font=font_xs,
            )
            draw.text((ax0 + 2, mid_y + 24), f"{sc:+.2f}", fill=sc_color, font=font_s)

    draw.text(
        (margin, y0 + cell_h + 10),
        "  ->  ".join(tids),
        fill=(30, 30, 30),
        font=font_xs,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def calibrate() -> tuple[dict, dict, list]:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_tids = list(gt["gt_tids"])
    tp.SIM_MIN = SIM_MIN
    mode = tp.configure_for_input(str(MERGE_DIR))
    tracks = tp.load_tracks(str(MERGE_DIR))
    print(f"[calibrate] mode={mode} tracks={len(tracks)} GT={len(gt_tids)} SIM_MIN={tp.SIM_MIN}")

    by_tid = {t.tid: t for t in tracks}
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    missing = [t for t in gt_tids if t not in by_tid]
    if missing:
        print("缺失 GT：", missing)

    excluded = list_same_cam_excluded(gt_tracks)
    print(f"[calibrate] emb_same 排除同鏡配對 {len(excluded)} 對：")
    for e in excluded:
        print(f"  {e['a']} ↔ {e['b']}  cam={e['cam']}  emb={e['emb']:.4f}")

    samples = tp.collect_gt_samples(tracks, gt_tids, removed_mislabel=[])
    calib = tp.fit_calibration(samples)
    calib["dataset"] = MERGE_DIR.name
    calib["input_dir"] = str(MERGE_DIR)
    calib["gt_path"] = str(GT_PATH.resolve())
    calib["meta"]["same_cam_excluded_pairs"] = excluded
    calib["meta"]["warning"] = "IN-SAMPLE：車輛校準與評估同一資料集 0507，僅供診斷"
    calib["meta"]["sim_min"] = SIM_MIN

    tp_prior = tp.compute_transition_prior(tracks, gt_tids)
    calib["transition_prior"] = tp_prior
    print(
        f"[calibrate] transition prior p_edge={tp_prior['p_edge']:.6f} = "
        f"{tp_prior['n_gt_true_transitions']}/{tp_prior['n_legal_edges']}"
    )

    pkl_path = OUT_ROOT / "calibration_vehicle_gt0507.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(calib, f)

    hist_png = OUT_ROOT / "emb_same_diff_hist_vehicle_gt0507.png"
    tp.save_emb_histogram(samples["emb_same"], samples["emb_diff"], hist_png)

    # 人員樣本直方圖對照（從人員 pkl 只有 fit 參數；重算人員 samples 太重——
    # 改從人員校準報告／重新載入人員 tracks 取 raw？為對照 μσ 夠用；直方圖並列需 raw。
    # 重跑人員 collect 以畫對照圖（SIM_MIN=0.85、人員 GT）。
    person_gt = json.loads(
        (OUT_ROOT / "ground_truth_20260507.json").read_text(encoding="utf-8")
    )
    person_merge = tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"
    tp.SIM_MIN = 0.85
    tp.configure_for_input(str(person_merge))
    person_tracks = tp.load_tracks(str(person_merge))
    person_samples = tp.collect_gt_samples(
        person_tracks,
        person_gt["person_tids"],
        removed_mislabel=list(person_gt.get("removed_mislabel") or ["K8-08_43"]),
    )
    compare_hist = OUT_ROOT / "emb_same_diff_hist_vehicle_vs_person_0507.png"
    save_compare_hist(
        samples["emb_same"],
        samples["emb_diff"],
        person_samples["emb_same"],
        person_samples["emb_diff"],
        compare_hist,
    )

    # 還原車輛模式給後續 run
    tp.SIM_MIN = SIM_MIN
    tp.configure_for_input(str(MERGE_DIR))

    person_calib = pickle.load(PERSON_CALIB.open("rb"))
    report = OUT_ROOT / "calibration_vehicle_gt0507_report.txt"
    write_vehicle_calib_report(
        calib,
        samples,
        excluded,
        person_calib,
        person_samples,
        hist_png,
        compare_hist,
        report,
    )
    print(f"寫入：{pkl_path}")
    print(f"寫入：{report}")
    print(f"寫入：{hist_png}")
    print(f"寫入：{compare_hist}")
    return calib, samples, gt_tracks


def write_vehicle_calib_report(
    calib,
    samples,
    excluded,
    person_calib,
    person_samples,
    hist_png,
    compare_hist,
    out_txt: Path,
) -> None:
    lines = []
    lines.append("=== calibration from vehicle GT 0507 ===")
    lines.append("WARNING: IN-SAMPLE（校準與評估同一資料集），結論僅供診斷")
    lines.append(f"SIM_MIN={SIM_MIN}")
    lines.append(f"counts: {calib['meta']['counts']}")
    lines.append(f"missing_gt: {calib['meta']['missing_gt']}")
    lines.append("")
    lines.append("--- emb_same：GT 兩兩跨鏡（已排除同鏡）---")
    lines.append(f"排除同鏡配對 n={len(excluded)}：")
    for e in excluded:
        lines.append(
            f"  EXCLUDED {e['a']} ↔ {e['b']}  cam={e['cam']}  emb={e['emb']:.4f}"
        )
    lines.append(str(calib["emb_same"]))
    lines.append("--- emb_diff：GT × 非GT 全配對 ---")
    lines.append(str(calib["emb_diff"]))
    lines.append("--- sim|GT / sim|nonGT ---")
    lines.append(str(calib.get("sim_gt")))
    lines.append(str(calib.get("sim_nongt")))
    lines.append("--- dH|same ---")
    lines.append(str(calib["dh_same"]))
    lines.append("")
    lines.append("=== 車 vs 人 並列（μ / σ / n）===")
    lines.append(
        f"{'dist':<12} {'vehicle μ':>10} {'vehicle σ':>10} {'veh n':>6}  "
        f"{'person μ':>10} {'person σ':>10} {'per n':>6}"
    )
    for key in ("emb_same", "emb_diff", "sim_gt", "sim_nongt"):
        v, p = calib[key], person_calib[key]
        lines.append(
            f"{key:<12} {v['mu']:10.4f} {v['sigma']:10.4f} {v['n']:6d}  "
            f"{p['mu']:10.4f} {p['sigma']:10.4f} {p['n']:6d}"
        )
    vs = _stats(samples["emb_same"])
    vd = _stats(samples["emb_diff"])
    ps = _stats(person_samples["emb_same"])
    pd = _stats(person_samples["emb_diff"])
    lines.append("")
    lines.append(
        f"預測核對：車 emb_same μ={vs['mu']:.4f} vs 人 {ps['mu']:.4f} "
        f"（車更高？ {vs['mu'] is not None and ps['mu'] is not None and vs['mu'] > ps['mu']}）"
    )
    lines.append(
        f"預測核對：車 emb_diff μ={vd['mu']:.4f} vs 人 {pd['mu']:.4f} "
        f"（diff 右移？ {vd['mu'] is not None and pd['mu'] is not None and vd['mu'] > pd['mu']}）"
    )
    lines.append("")
    tp_prior = calib.get("transition_prior") or {}
    lines.append("--- transition prior ---")
    lines.append(
        f"  n_gt_true_transitions={tp_prior.get('n_gt_true_transitions')}  "
        f"n_legal_edges={tp_prior.get('n_legal_edges')}  "
        f"p_edge={tp_prior.get('p_edge')}"
    )
    for e in tp_prior.get("gt_transitions") or []:
        lines.append(
            f"  {e['from']} -> {e['to']} via {e['via']}  "
            f"dt={e['dt']:.2f} hop={e['hop']} emb={e['emb']:.3f}"
        )
    lines.append("")
    lines.append(f"histogram: {hist_png}")
    lines.append(f"compare histogram: {compare_hist}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_m0(calib_path: Path) -> dict:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # 凍結 M0：與人員 0507_top1.json options 對齊；僅換校準表
    argv = [
        str(MERGE_DIR),
        "--sim-min",
        str(SIM_MIN),
        "--calibration",
        str(calib_path),
        "--out-dir",
        str(RUN_DIR),
        "--dt-scoring",
        "off",
        "--transition-prior",
        "off",
    ]
    print("[run] M0 frozen:", " ".join(argv))
    summary = tp.cmd_run(argv)
    return summary


def parse_top1_from_out_txt(out_txt: Path) -> tuple[list[dict], list[dict]]:
    """從 0507_out.txt 解析 Top-1 的 NODE / 邊帳。"""
    text = out_txt.read_text(encoding="utf-8")
    nodes, edges = [], []
    in_top1 = False
    for line in text.splitlines():
        if line.startswith("#1  "):
            in_top1 = True
            continue
        if in_top1 and line.startswith("#2  "):
            break
        if not in_top1:
            continue
        s = line.strip()
        if s.startswith("NODE "):
            # NODE K8-28_2 sim=0.934  LLR_raw=+2.453 w=0.545  score=+1.338
            parts = s.split()
            tid = parts[1]
            kv = {}
            for p in parts[2:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k] = float(v)
            nodes.append(
                {
                    "super": tid,
                    "sim": kv.get("sim"),
                    "raw": kv.get("LLR_raw"),
                    "w": kv.get("w"),
                    "score": kv.get("score"),
                    "enabled": True,
                }
            )
        elif " -> " in s and "hop=" in s:
            # K8-28_2 -> K8-15_7   hop=1  dt=0.0s  ...
            left, rest = s.split("hop=", 1)
            fr, to = [x.strip() for x in left.split("->")]
            fields = {"from": fr, "to": to, "from_super": fr, "to_super": to}
            # tokenize rest
            chunk = ("hop=" + rest).replace("  ", " ")
            for token in chunk.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                if k == "dt":
                    fields["dt"] = float(v.rstrip("s"))
                elif k == "hop":
                    fields["hop"] = int(v)
                elif k == "emb":
                    fields["emb"] = float(v)
                elif k == "LLR_emb":
                    fields["LLR_emb"] = float(v)
                elif k == "LLR_dH":
                    fields["LLR_dH"] = None if v in ("—", "-", "n/a") else float(v)
                elif k == "LLR_dt":
                    fields["LLR_dt"] = None if v in ("removed", "n/a", "—") else float(v)
                elif k == "edge":
                    fields["score"] = float(v)
                elif k == "d_H":
                    fields["h_dist"] = None if v in ("—", "-") else float(v)
            edges.append(fields)
    return nodes, edges


def write_run_report(summary: dict | None, gt_tracks: list, gt_tids: list[str]) -> Path:
    gt_set = set(gt_tids)

    top_json = RUN_DIR / "0507_top1.json"
    data = json.loads(top_json.read_text(encoding="utf-8"))
    top1 = dict(data.get("top1") or {})
    nodes, edges = parse_top1_from_out_txt(RUN_DIR / "0507_out.txt")
    top1["edges"] = edges
    top1["node_evidence"] = nodes

    tids = list(top1.get("tids") or [])
    n = len(tids)
    hit = sum(1 for t in tids if t in gt_set)
    prec = hit / n if n else 0.0
    rec = hit / len(gt_set) if gt_set else 0.0
    pprob = float(top1.get("path_probability") or 0.0)

    collage = RUN_DIR / "0507_top1_gt_collage.png"
    if tids:
        render_gt_colored_collage(MERGE_DIR, top1, gt_set, collage)

    # 規模
    n_tracks = int(data.get("n_tracks") or 0)
    n_legal = int(data.get("n_legal_edges") or 0)
    enum = {}
    srep = data.get("supernodes") or {}
    enum = srep.get("enumeration") or {}
    if not enum:
        super_path = RUN_DIR / "0507_supernodes.json"
        if super_path.is_file():
            enum = (
                json.loads(super_path.read_text(encoding="utf-8")).get("enumeration")
                or {}
            )
    beam_mode = enum.get("mode")

    # GT dt 分布
    dt_rows = gt_true_transition_dts(gt_tracks)
    dts = [r["dt"] for r in dt_rows]
    exceed = [r for r in dt_rows if r["exceeds_dt_max"]]

    md = OUT_ROOT / "vehicle_0507_m0_report.md"
    lines = []
    lines.append("# 車輛 0507：校準 + M0 首跑報告")
    lines.append("")
    lines.append("設定全凍結（人員 M0 對齊）：`SIM_MIN=0.85`、emb gate fix on、supernode on、")
    lines.append("node evidence on、`dt_scoring=off`、`transition_prior=off`、`DT_MAX=130` 不動。")
    lines.append("僅換校準表：`calibration_vehicle_gt0507.pkl`。")
    lines.append("")
    lines.append("## 1. Top-1")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| path | `{' -> '.join(tids)}` |")
    lines.append(f"| precision | {prec:.4f} ({hit}/{n}) |")
    lines.append(f"| recall | {rec:.4f} ({hit}/{len(gt_set)}) |")
    lines.append(f"| P | {pprob:.6f} |")
    lines.append(f"| score | {top1.get('score')} |")
    lines.append(f"| n_segments | {top1.get('n_segments', 1)} |")
    fp = [t for t in tids if t not in gt_set]
    fn = sorted(gt_set - set(tids))
    lines.append(f"| hit | {', '.join(t for t in tids if t in gt_set) or '—'} |")
    lines.append(f"| FP | {', '.join(fp) or '—'} |")
    lines.append(f"| FN | {', '.join(fn) or '—'} |")
    lines.append("")
    lines.append(f"- 拼圖（GT 綠／非GT 紅）：`{collage}`")
    lines.append(f"- 文字邊帳：`{RUN_DIR / '0507_out.txt'}`")
    lines.append("")
    lines.append("## 2. 逐邊帳目（Top-1）")
    lines.append("")
    lines.append("| from → to | hop | dt | emb | LLR_emb | LLR_dt | LLR_dH | edge | ends GT? |")
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---|")
    for e in edges:
        fr = e.get("from_super") or e.get("from")
        to = e.get("to_super") or e.get("to")
        gt_tag = ("G" if fr in gt_set else "·") + ("G" if to in gt_set else "·")
        ldh = e.get("LLR_dH")
        ldh_s = f"{ldh:+.3f}" if ldh is not None else "—"
        ldt = e.get("LLR_dt")
        ldt_s = "removed/off" if ldt is None else f"{float(ldt):+.3f}"
        lines.append(
            f"| `{fr}` → `{to}` | {e.get('hop')} | {float(e.get('dt') or 0):.1f} | "
            f"{float(e.get('emb') or 0):.3f} | {float(e.get('LLR_emb') or 0):+.3f} | "
            f"{ldt_s} | {ldh_s} | {float(e.get('score') or 0):+.3f} | {gt_tag} |"
        )
    lines.append("")
    lines.append("節點證據：")
    lines.append("")
    for ne in nodes:
        sid = ne.get("super")
        tag = "GT" if sid in gt_set else "nonGT"
        lines.append(
            f"- `{sid}` ({tag}) sim={ne.get('sim'):.3f}  "
            f"LLR_raw={ne.get('raw'):+.3f} w={ne.get('w'):.3f}  "
            f"score={ne.get('score'):+.3f}"
        )
    lines.append("")
    lines.append("## 3. 規模健檢")
    lines.append("")
    lines.append(f"| item | value |")
    lines.append(f"|---|---|")
    lines.append(f"| 候選 track 數 | {n_tracks or data.get('n_tracks')} |")
    lines.append(f"| 合法邊數 | {n_legal or enum.get('n_legal_edges')} |")
    lines.append(f"| FULL_ENUM_EDGE_CAP | {tp.FULL_ENUM_EDGE_CAP} |")
    lines.append(f"| enumeration mode | {enum.get('mode') or beam_mode or '？'} |")
    lines.append(f"| beam_width | {enum.get('beam_width')} |")
    lines.append(f"| beam_max_leaves | {enum.get('beam_max_leaves')} |")
    lines.append(f"| n_beam_leaves / paths | {enum.get('n_beam_leaves') or data.get('n_paths')} |")
    triggered = (enum.get("mode") == "beam") or (
        (n_legal or enum.get("n_legal_edges") or 0) > tp.FULL_ENUM_EDGE_CAP
    )
    lines.append(f"| **是否觸發 beam** | **{'是' if triggered else '否'}** |")
    lines.append("")
    lines.append(
        f"（車輛舊帳曾路徑爆炸；本跑合法邊="
        f"{n_legal or enum.get('n_legal_edges')}，cap={tp.FULL_ENUM_EDGE_CAP}）"
    )
    lines.append("")
    lines.append("## 4. 車輛 GT 真轉移 dt 分布（凍結 DT_MAX=130，僅報告）")
    lines.append("")
    lines.append(f"相鄰 GT 對數（t_start 排序）= {len(dt_rows)}")
    if dts:
        lines.append(
            f"- n={len(dts)}  μ={statistics.mean(dts):.2f}  "
            f"σ={statistics.pstdev(dts) if len(dts) > 1 else 0:.2f}  "
            f"min={min(dts):.2f}  median={statistics.median(dts):.2f}  "
            f"max={max(dts):.2f}"
        )
        q = np.quantile(np.asarray(dts), [0.25, 0.5, 0.75, 0.9, 0.95])
        lines.append(
            f"- quantiles p25/p50/p75/p90/p95 = "
            f"{q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}/{q[3]:.1f}/{q[4]:.1f}"
        )
    lines.append(f"- DT_MAX={tp.DT_MAX}")
    lines.append(f"- 超出 DT_MAX 的真轉移：{len(exceed)} 條")
    if exceed:
        lines.append("")
        lines.append("| from → to | dt | hop | edge_ok | reason |")
        lines.append("|---|---:|---:|---|---|")
        for r in exceed:
            lines.append(
                f"| `{r['from']}` → `{r['to']}` | {r['dt']:.1f} | {r['hop']} | "
                f"{r['edge_ok']} | {r['edge_reason']} |"
            )
    else:
        lines.append("- （無超出 DT_MAX 的相鄰真轉移）")
    lines.append("")
    lines.append("全表（含未超 DT_MAX）：")
    lines.append("")
    lines.append("| from → to | dt_raw | dt | hop | edge_ok | reason | emb |")
    lines.append("|---|---:|---:|---:|---|---|---:|")
    for r in dt_rows:
        emb_s = f"{r['emb']:.3f}" if r["emb"] is not None else "—"
        lines.append(
            f"| `{r['from']}` → `{r['to']}` | {r['dt_raw']:.1f} | {r['dt']:.1f} | "
            f"{r['hop']} | {r['edge_ok']} | {r['edge_reason'] or 'ok'} | {emb_s} |"
        )
    lines.append("")
    lines.append("## 5. 產出檔案")
    lines.append("")
    for p in (
        OUT_ROOT / "calibration_vehicle_gt0507.pkl",
        OUT_ROOT / "calibration_vehicle_gt0507_report.txt",
        OUT_ROOT / "emb_same_diff_hist_vehicle_gt0507.png",
        OUT_ROOT / "emb_same_diff_hist_vehicle_vs_person_0507.png",
        RUN_DIR / "0507_out.txt",
        RUN_DIR / "0507_top1.json",
        collage,
        md,
    ):
        lines.append(f"- `{p}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"寫入：{md}")
    return md


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--report-only",
        action="store_true",
        help="跳過校準／M0，只依既有產出寫報告",
    )
    args = p.parse_args()
    calib_path = OUT_ROOT / "calibration_vehicle_gt0507.pkl"
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    if not args.report_only:
        calibrate()
        run_m0(calib_path)

    tp.SIM_MIN = SIM_MIN
    tp.configure_for_input(str(MERGE_DIR))
    tracks = tp.load_tracks(str(MERGE_DIR))
    by_tid = {t.tid: t for t in tracks}
    gt_tracks = [by_tid[t] for t in gt["gt_tids"] if t in by_tid]
    write_run_report(None, gt_tracks, gt["gt_tids"])
    print("DONE")


if __name__ == "__main__":
    main()
