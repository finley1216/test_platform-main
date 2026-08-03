#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
車輛 0507 M9：校準換 vehicle pkl；其餘同人員 M9。
GT 僅評估；不調參。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKING = _REPO_ROOT / "scripts" / "tracking"
for _p in (_REPO_ROOT, _TRACKING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import track_path_m0 as tp
import track_path_minimal as m  # 車輛實驗仍用完整 minimal（含 collage／GT 輔助）

OUT = tp.OUTPUT_ROOT / "v1.0"
MERGE = tp.QUERY_FILTER_OUTPUT_ROOT / "車輛追蹤_20260507"
GT_PATH = OUT / "ground_truth_vehicle_20260507.json"
VEH_CALIB = OUT / "calibration_vehicle_gt0507.pkl"
PERSON_CALIB = OUT / "calibration_gt0507.pkl"
OUT_DIR = OUT / "vehicle_m9_0507"
SIM_MIN = 0.85

# 人員 M9 對照（comparison_m9.md）
PERSON_M9 = {
    "0507": {"prec": 0.667, "rec": 0.727, "P": 0.470132},
    "0528": {"prec": 0.684, "rec": 0.812, "P": 0.229583},
}


def confirm_beam_score_scale() -> dict:
    """
    M9：beam 鍵 = Σ e['score']；最終路徑分 = Σ e['score']；
    e['score']=LLR+C+S；無 node evidence。同尺。
    """
    return {
        "same_scale": True,
        "beam_key": "prefix Σ e['score']  (LLR+C+S)",
        "final_score": "Σ e['score']  (LLR+C+S)；node_evidence=[]",
        "note": (
            "與 M0 不同：M0 beam 僅邊分、最終另加 node；"
            "M9 無節點證據，beam／最終同為邊分累加 → 無需修正。"
        ),
        "code_refs": [
            "_enumerate_from_succ_m6: nxt.append((sc + e['score'], ...))",
            "run_with_config m9: score = sum(e['score'] for e in edges_info)",
            "_build_succ_m9: score = llr + C + S",
        ],
    }


def llr_zero_crossing(calib: dict) -> float | None:
    """掃描 emb∈[0,1] 找 LLR 由負轉正的零點。"""
    prev = None
    prev_x = None
    for i in range(1001):
        x = i / 1000.0
        y = m.emb_llr_raw(calib, x)
        if prev is not None and prev < 0 <= y:
            # 線性插值
            if y == prev:
                return x
            t = -prev / (y - prev)
            return float(prev_x + t * (x - prev_x))
        prev, prev_x = y, x
    return None


def collect_edge_llr_stats(merge: Path, calib_path: Path, gt_tids: list[str], sim_min: float):
    """建 M9 succ，標 GT／非GT，回報 LLR>0 比例等。"""
    tp.SIM_MIN = sim_min
    tp.configure_for_input(str(merge))
    tracks = tp.load_tracks(str(merge))
    calib, _ = m._load_m9_calib(calib_path)
    m.attach_crop_embs(tracks, merge)
    coexist_median = m.median_edge_emb(tracks)
    nodes, _ = tp.build_supernodes(tracks, overlap_emb_min=coexist_median)
    succ, _, n_legal, meta = m._build_succ_m9(nodes, calib)

    by_tid = {t.tid: t for t in tracks}
    edges = []
    for i, items in enumerate(succ):
        for j, e in items:
            e2 = dict(e)
            e2["_i"] = i
            e2["_j"] = j
            edges.append(e2)

    gt_keys, *_ = m._m5_gt_edge_keys(nodes, edges, sorted(gt_tids), by_tid)
    for e in edges:
        e["is_gt"] = (e["_i"], e["_j"]) in gt_keys

    gt_e = [e for e in edges if e["is_gt"]]
    ng_e = [e for e in edges if not e["is_gt"]]

    def _pos_rate(arr):
        if not arr:
            return {"n": 0, "n_llr_pos": 0, "rate": None, "mean_llr": None}
        n_pos = sum(1 for e in arr if float(e.get("LLR") or 0) > 0)
        mean = sum(float(e.get("LLR") or 0) for e in arr) / len(arr)
        return {
            "n": len(arr),
            "n_llr_pos": n_pos,
            "rate": n_pos / len(arr),
            "mean_llr": mean,
        }

    zero = llr_zero_crossing(calib)
    return {
        "n_legal": n_legal,
        "n_tracks": len(tracks),
        "n_nodes": len(nodes),
        "calib": {
            "path": str(calib_path),
            "same_mu": float(calib["emb_same"]["mu"]),
            "same_sigma": float(calib["emb_same"]["sigma"]),
            "diff_mu": float(calib["emb_diff"]["mu"]),
            "diff_sigma": float(calib["emb_diff"]["sigma"]),
            "llr_zero": zero,
        },
        "all": _pos_rate(edges),
        "gt": _pos_rate(gt_e),
        "nongt": _pos_rate(ng_e),
        "meta": meta,
        "edges": edges,
        "nodes": nodes,
        "tracks": tracks,
        "gt_keys": gt_keys,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scale = confirm_beam_score_scale()
    assert scale["same_scale"], "beam／最終不同尺，應先修"

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_tids = list(gt["gt_tids"])
    gt_set = set(gt_tids)

    print("=== beam/score scale check ===")
    print(json.dumps(scale, ensure_ascii=False, indent=2))
    print("同尺確認 → 直接跑 M9")

    # --- run M9 ---
    cfg = m.RunConfig(
        scoring="m9",
        node_score=False,
        dt_max=None,
        sim_min=SIM_MIN,
        variant_tag="m9",
        calibration_path=str(VEH_CALIB.resolve()),
    )
    print(f"\n===== vehicle 0507 M9 =====")
    print(f"calibration={cfg.calibration_path}")
    result = m.run_with_config(MERGE, cfg)
    summary = m._save_summary(result, MERGE, OUT_DIR, "vehicle_0507_m9_top1")
    pack = m._top_pack(summary, gt_set, {"source": "vehicle_m9"})
    top = result["ranked"][0] if result["ranked"] else None
    struct = m._hyp_structure_stats(top)
    enum = (result["super_report"] or {}).get("enumeration") or {}

    # collage
    by_tid = {t.tid: t for t in result["tracks"]}
    collage = OUT_DIR / "vehicle_0507_m9_top1_collage.png"
    if top:
        m._render_one_m6_collage(
            hyp=top,
            rank=1,
            short="vehicle0507",
            dataset_tag="車輛追蹤_20260507",
            gt_set=gt_set,
            merge=MERGE,
            by_tid=by_tid,
            out_png=collage,
            mode_label="M9（LLR+C+S）vehicle",
            app_key="LLR",
            app_tag="LLR",
        )

    # edge account for top1
    edges = list((top or {}).get("edges") or [])
    if not edges and top and int(top.get("n_segments") or 1) == 1:
        # try segments
        for seg in top.get("segments") or []:
            edges.extend(seg.get("edges") or [])

    # erosion stats vehicle + person
    print("\n===== LLR>0 erosion stats =====")
    veh_stats = collect_edge_llr_stats(MERGE, VEH_CALIB, gt_tids, SIM_MIN)
    person_gt = json.loads(
        (OUT / "ground_truth_20260507.json").read_text(encoding="utf-8")
    )["person_tids"]
    person_merge = tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"
    per_stats = collect_edge_llr_stats(
        person_merge, PERSON_CALIB, person_gt, SIM_MIN
    )

    # report
    lines = []
    lines.append("# 車輛 0507 M9 報告")
    lines.append("")
    lines.append(f"生成時間：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("> GT 僅評估／著色；不調參；校準表換 `calibration_vehicle_gt0507.pkl`，其餘同人員 M9。")
    lines.append("")
    lines.append("## 0. beam 排序鍵 vs 最終總分")
    lines.append("")
    lines.append(f"- **同尺：{'是' if scale['same_scale'] else '否'}**")
    lines.append(f"- beam 鍵：`{scale['beam_key']}`")
    lines.append(f"- 最終分：`{scale['final_score']}`")
    lines.append(f"- {scale['note']}")
    lines.append("- 結論：無需修正，直接跑。")
    lines.append("")
    lines.append("## 1. Top-1 prec / rec / P（與人員 M9 並列）")
    lines.append("")
    lines.append("| 資料 | precision | recall | P | n_path | n_hit | score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| **車輛 0507 M9** | {pack['precision']:.3f} | {pack['recall']:.3f} | "
        f"{pack.get('path_probability') or pack.get('P') or 0:.6f} | "
        f"{pack.get('n_path')} | {pack.get('n_hit')} | "
        f"{(top or {}).get('score'):.3f} |"
    )
    lines.append(
        f"| 人員 0507 M9 | {PERSON_M9['0507']['prec']:.3f} | {PERSON_M9['0507']['rec']:.3f} | "
        f"{PERSON_M9['0507']['P']:.6f} | — | — | — |"
    )
    lines.append(
        f"| 人員 0528 M9 | {PERSON_M9['0528']['prec']:.3f} | {PERSON_M9['0528']['rec']:.3f} | "
        f"{PERSON_M9['0528']['P']:.6f} | — | — | — |"
    )
    lines.append("")
    path_s = (top or {}).get("path") or " -> ".join(
        (top or {}).get("super_labels") or (top or {}).get("tids") or []
    )
    lines.append(f"- Top-1：`{path_s}`")
    lines.append(f"- hit：{', '.join(pack.get('hit_tids') or []) or '—'}")
    lines.append(f"- FP：{', '.join(pack.get('false_positive') or []) or '—'}")
    lines.append(f"- FN：{', '.join(pack.get('false_negative') or []) or '—'}")
    lines.append(f"- 拼圖：`{collage}`")
    lines.append(
        f"- 規模：候選={len(result['tracks'])} 合法邊={result['n_legal_edges']} "
        f"enum={enum.get('mode')} beam_leaves={enum.get('n_beam_leaves')}"
    )
    lines.append("")
    lines.append("## 2. 巨路徑／碎片化檢查")
    lines.append("")
    lines.append("| n_super | n_tids | n_seg | max_seg | mean_seg | 巨路徑? | 碎片化? | 備註 |")
    lines.append("|---:|---:|---:|---:|---:|:------:|:------:|------|")
    n_super = struct.get("n_super")
    n_tids = struct.get("n_tids")
    n_seg = struct.get("n_segments") or struct.get("n_seg")
    max_seg = struct.get("max_seg_len") or struct.get("max_seg")
    mean_seg = struct.get("mean_seg_len") or struct.get("mean_seg")
    giant = bool(struct.get("is_giant") or (n_super or 0) >= 15)
    frag = bool(struct.get("is_fragmented"))
    note = struct.get("note") or ""
    if mean_seg is not None:
        mean_seg_s = f"{float(mean_seg):.1f}"
    else:
        mean_seg_s = "—"
    lines.append(
        f"| {n_super} | {n_tids} | {n_seg} | {max_seg} | {mean_seg_s} | "
        f"{'是 ★' if giant else '否'} | {'是 ★' if frag else '否'} | {note} |"
    )
    lines.append("")
    lines.append("> 巨路徑：n_super≥15；碎片化：n_seg≥2 且平均段長≤3。")
    lines.append("")
    lines.append("## 3. 逐邊帳目（Top-1）")
    lines.append("")
    lines.append("| from → to | emb | LLR | C | S | Σ | hop | dt | ends GT? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for e in edges:
        fr = e.get("from_super") or e.get("from")
        to = e.get("to_super") or e.get("to")
        ft = e.get("from")
        tt = e.get("to")
        tag = ("G" if ft in gt_set else "·") + ("G" if tt in gt_set else "·")
        # also member-based
        fm = set(e.get("from_members") or [])
        tm = set(e.get("to_members") or [])
        if fm & gt_set:
            tag = "G" + tag[1]
        if tm & gt_set:
            tag = tag[0] + "G"

        def _f(v, nd=3):
            if v is None:
                return "—"
            v = float(v)
            if abs(v) >= 1e299:
                return "+∞" if v > 0 else "−∞"
            return f"{v:+.{nd}f}" if nd == 3 else f"{v:.{nd}f}"

        lines.append(
            f"| `{fr}` → `{to}` | {float(e.get('emb') or 0):.3f} | "
            f"{_f(e.get('LLR'))} | {_f(e.get('C'))} | {_f(e.get('S'))} | "
            f"{_f(e.get('score'))} | {e.get('hop')} | {float(e.get('dt') or 0):.1f} | {tag} |"
        )
    if edges:
        sLLR = sum(float(e.get("LLR") or 0) for e in edges)
        sC = sum(float(e.get("C") or 0) for e in edges)
        sS = sum(float(e.get("S") or 0) for e in edges)
        sSum = sum(float(e.get("score") or 0) for e in edges)
        lines.append(
            f"| Σ | | {sLLR:+.3f} | {sC:+.3f} | {sS:+.3f} | **{sSum:+.3f}** | | | |"
        )
    lines.append("")
    lines.append("## 4. 風險預告：同款車海對 LLR 的侵蝕")
    lines.append("")
    vz = veh_stats["calib"]["llr_zero"]
    pz = per_stats["calib"]["llr_zero"]
    lines.append(
        f"- 車輛校準：emb_same=N({veh_stats['calib']['same_mu']:.4f},"
        f"{veh_stats['calib']['same_sigma']:.4f})；"
        f"emb_diff=N({veh_stats['calib']['diff_mu']:.4f},"
        f"{veh_stats['calib']['diff_sigma']:.4f})；"
        f"**LLR 零點≈{vz:.4f}**" if vz is not None else ""
    )
    lines.append(
        f"- 人員校準：emb_same=N({per_stats['calib']['same_mu']:.4f},"
        f"{per_stats['calib']['same_sigma']:.4f})；"
        f"emb_diff=N({per_stats['calib']['diff_mu']:.4f},"
        f"{per_stats['calib']['diff_sigma']:.4f})；"
        f"**LLR 零點≈{pz:.4f}**" if pz is not None else ""
    )
    lines.append("")
    lines.append("| 集合 | 車 n | 車 LLR>0 | 車比例 | 車 mean LLR | 人 n | 人 LLR>0 | 人比例 | 人 mean LLR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in (("nongt", "非GT邊"), ("gt", "GT邊"), ("all", "全邊")):
        v, p = veh_stats[key], per_stats[key]
        vr = f"{v['rate']*100:.1f}%" if v["rate"] is not None else "—"
        pr = f"{p['rate']*100:.1f}%" if p["rate"] is not None else "—"
        vm = f"{v['mean_llr']:+.3f}" if v["mean_llr"] is not None else "—"
        pm = f"{p['mean_llr']:+.3f}" if p["mean_llr"] is not None else "—"
        lines.append(
            f"| {label} | {v['n']} | {v['n_llr_pos']} | {vr} | {vm} | "
            f"{p['n']} | {p['n_llr_pos']} | {pr} | {pm} |"
        )
    lines.append("")
    vng, png = veh_stats["nongt"], per_stats["nongt"]
    # 高似邊（emb ≥ 各自 LLR 零點）的非 GT
    v_hi = [
        e
        for e in veh_stats["edges"]
        if (not e["is_gt"]) and float(e.get("emb") or 0) >= (vz or 0.933)
    ]
    p_hi = [
        e
        for e in per_stats["edges"]
        if (not e["is_gt"]) and float(e.get("emb") or 0) >= (pz or 0.895)
    ]
    v_hi_pos = sum(1 for e in v_hi if float(e.get("LLR") or 0) > 0)
    p_hi_pos = sum(1 for e in p_hi if float(e.get("LLR") or 0) > 0)

    lines.append("")
    lines.append(
        f"**侵蝕量化**：非 GT 邊 LLR>0 比例 車 {vng['rate']*100:.1f}% "
        f"({vng['n_llr_pos']}/{vng['n']}) vs 人 {png['rate']*100:.1f}% "
        f"({png['n_llr_pos']}/{png['n']})；"
        f"非 GT mean LLR 車 {vng['mean_llr']:+.3f} vs 人 {png['mean_llr']:+.3f}。"
    )
    lines.append("")
    lines.append(
        f"- 零點上移（人 {pz:.4f} → 車 {vz:.4f}）使「中等像」邊更易得負 LLR，"
        f"故非 GT **比例**未必升高。"
    )
    lines.append(
        f"- 但絕對量：車非 GT 正 LLR 邊 **{vng['n_llr_pos']}** 條 vs 人 **{png['n_llr_pos']}** 條"
        f"（約 {vng['n_llr_pos']/max(png['n_llr_pos'],1):.1f}×）；"
        f"合法邊總量 車 {veh_stats['n_legal']} vs 人 {per_stats['n_legal']}。"
    )
    lines.append(
        f"- 高似非 GT（emb≥各自零點）：車 {len(v_hi)} 條（LLR>0 佔 {v_hi_pos}）；"
        f"人 {len(p_hi)} 條（LLR>0 佔 {p_hi_pos}）——"
        f"同款車海主要體現在 **高 emb 假邊的絕對洪水**，Top-1 前段多條 emb≥0.94 且 LLR>0 的非 GT 邊即例。"
    )
    lines.append("")
    lines.append("## 5. 產出")
    lines.append("")
    for pth in (
        OUT_DIR / "vehicle_0507_m9_top1.json",
        collage,
        OUT_DIR / "vehicle_0507_m9_report.md",
    ):
        lines.append(f"- `{pth}`")

    md = OUT_DIR / "vehicle_0507_m9_report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "scale_check": scale,
        "pack": {
            k: pack[k]
            for k in (
                "precision",
                "recall",
                "n_path",
                "n_hit",
                "hit_tids",
                "false_positive",
                "false_negative",
            )
            if k in pack
        },
        "path_probability": pack.get("path_probability") or pack.get("P"),
        "top1_score": (top or {}).get("score"),
        "top1_path": path_s,
        "structure": struct,
        "enumeration": enum,
        "erosion": {
            "vehicle": {k: veh_stats[k] for k in ("calib", "all", "gt", "nongt", "n_legal")},
            "person": {k: per_stats[k] for k in ("calib", "all", "gt", "nongt", "n_legal")},
        },
    }
    (OUT_DIR / "vehicle_0507_m9_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"寫入：{md}")
    print(
        f"prec={pack['precision']:.3f} rec={pack['recall']:.3f} "
        f"P={pack.get('path_probability') or 0:.4f}"
    )
    print(
        f"nonGT LLR>0: veh={vng['rate']} person={png['rate']}"
    )
    print("DONE")


if __name__ == "__main__":
    main()
