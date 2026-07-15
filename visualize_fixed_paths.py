# -*- coding: utf-8 -*-
"""
修正後路徑視覺化（不改演算法／計分，僅輸出圖與 HTML）
======================================================
以 gt_calib_0507_fixed 最終結果為準。
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import path_enum_llr as llr  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

COLOR_GT = (34, 139, 34)
COLOR_NON_GT = (200, 40, 40)
COLOR_UNK = (120, 120, 120)
COLOR_HI = (220, 140, 0)  # 09_42 強調
BG = (252, 252, 252)

EXPECTED_GT_SUPER = [
    ["K8-09_7"],
    ["K8-08_30", "K8-01_7"],
    ["K8-07_40"],
    ["K8-23_8", "K8-22_19"],
    ["K8-22_22"],
    ["K8-07_112"],
    ["K8-01_50"],
    ["K8-08_77", "K8-01_62"],
]


def _font(size: int, bold: bool = False):
    cands = [
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in cands:
        p = Path(name)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (236, 236, 236))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def _pick_three(crops: list[Path]) -> list[Path | None]:
    if not crops:
        return [None, None, None]
    if len(crops) == 1:
        return [crops[0], crops[0], crops[0]]
    if len(crops) == 2:
        return [crops[0], crops[0], crops[1]]
    return [crops[0], crops[len(crops) // 2], crops[-1]]


def load_gt_set(gt_path: Path) -> set[str]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    return set(data.get("person_tids") or [])


def border_color(tid: str, gt_set: set[str] | None) -> tuple[int, int, int]:
    if gt_set is None:
        return COLOR_UNK
    return COLOR_GT if tid in gt_set else COLOR_NON_GT


def parse_super_columns(path: dict) -> list[dict]:
    """回傳 columns: [{label, members:[tid,...]}]，優先用 edges 的 members。"""
    edges = path.get("edges") or []
    labels = path.get("super_labels")
    tids = path.get("tids") or []

    if edges and edges[0].get("from_members") is not None:
        cols = [{"label": edges[0].get("from_super") or edges[0]["from"], "members": list(edges[0]["from_members"])}]
        for e in edges:
            cols.append(
                {
                    "label": e.get("to_super") or e["to"],
                    "members": list(e.get("to_members") or [e["to"]]),
                }
            )
        return cols

    if labels:
        cols = []
        for lab in labels:
            if lab.startswith("{") and lab.endswith("}"):
                members = [x.strip() for x in lab[1:-1].split(",") if x.strip()]
            else:
                members = [lab]
            cols.append({"label": lab, "members": members})
        return cols

    return [{"label": t, "members": [t]} for t in tids]


def track_meta(by_tid: dict, tid: str) -> dict:
    t = by_tid.get(tid)
    if t is None:
        cam, tid_s = tid.rsplit("_", 1)
        return {
            "tid": tid,
            "cam": cam,
            "t_start": None,
            "t_end": None,
            "sim": None,
        }
    return {
        "tid": tid,
        "cam": t.cam,
        "t_start": float(t.t_start),
        "t_end": float(t.t_end),
        "sim": float(t.sim),
    }


def load_crops(merge_dir: Path, tid: str) -> list[Path]:
    cam, tid_s = tid.rsplit("_", 1)
    try:
        _, crops = pes._crop_paths_for_track(merge_dir, cam, int(tid_s))
        return crops
    except Exception:
        return []


def reconstruct_edges_for_columns(
    columns: list[dict],
    by_tid: dict,
    calib: dict,
) -> list[dict]:
    """依欄位成員重算邊（僅顯示用，公式同 path_enum_llr.edge_llr）。"""
    edges = []
    from path_enum_llr import SuperNode, _best_member_edge, edge_llr

    def sn_from_members(members: list[str]):
        tracks = [by_tid[m] for m in members if m in by_tid]
        if not tracks:
            return None
        emb = np.stack([t.emb for t in tracks]).mean(0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return SuperNode(
            sid="+".join(members),
            members=tracks,
            emb=emb,
            sim=float(np.mean([t.sim for t in tracks])),
            t_start=min(t.t_start for t in tracks),
            t_end=max(t.t_end for t in tracks),
            cams=sorted({t.cam for t in tracks}),
        )

    for i in range(len(columns) - 1):
        sa = sn_from_members(columns[i]["members"])
        sb = sn_from_members(columns[i + 1]["members"])
        if sa is None or sb is None:
            edges.append({"hop": None, "dt": None, "score": None, "missing": True})
            continue
        best, _ = _best_member_edge(sa, sb)
        if best is None:
            edges.append({"hop": None, "dt": None, "score": None, "missing": True})
            continue
        u, v, dt, hop, emb, h_dist = best
        e = edge_llr(u, v, dt, emb, h_dist, calib)
        e["hop"] = hop
        e["from_super"] = columns[i]["label"]
        e["to_super"] = columns[i + 1]["label"]
        e["from_members"] = columns[i]["members"]
        e["to_members"] = columns[i + 1]["members"]
        edges.append(e)
    return edges


def render_path_collage(
    merge_dir: Path,
    path: dict,
    by_tid: dict,
    out_png: Path,
    *,
    title: str,
    gt_set: set[str] | None,
    highlight_tid: str | None = "K8-09_42",
    calib: dict | None = None,
) -> Path:
    columns = parse_super_columns(path)
    edges = path.get("edges") or []
    # 若 edges 數與欄數不一致，重算
    if len(edges) != max(0, len(columns) - 1) and calib is not None:
        edges = reconstruct_edges_for_columns(columns, by_tid, calib)
    elif edges and edges[0].get("from_members") is None and calib is not None:
        edges = reconstruct_edges_for_columns(columns, by_tid, calib)

    # layout sizes
    thumb = (96, 128)
    thumb_hi = (128, 170)
    member_pad = 6
    label_h = 36
    meta_h = 52
    gap_w = 88
    margin = 18
    title_h = 52
    foot_h = 40

    # per-column width / height
    col_infos = []
    for col in columns:
        hi = highlight_tid is not None and highlight_tid in col["members"]
        tw, th = (thumb_hi if hi else thumb)
        n_mem = max(1, len(col["members"]))
        # 3 thumbs stacked per member + labels
        mem_h = 3 * (th + 4) + meta_h + (14 if hi else 0)
        col_h = label_h + n_mem * (mem_h + member_pad) + 8
        col_w = max(tw + 24, 168 if not hi else 210)
        if hi:
            col_w = max(col_w, 230)
        col_infos.append({"w": col_w, "h": col_h, "tw": tw, "th": th, "hi": hi})

    body_h = max(c["h"] for c in col_infos) + 20
    width = margin * 2 + sum(c["w"] for c in col_infos) + gap_w * max(0, len(columns) - 1)
    height = title_h + body_h + foot_h + margin
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    font_t = _font(17)
    font_s = _font(12)
    font_xs = _font(10)
    font_b = _font(13, bold=True)

    score = path.get("score")
    pprob = path.get("path_probability")
    title_line = title
    if score is not None:
        title_line += f"  score={score:.3f}"
    if pprob is not None:
        title_line += f"  P={pprob:.4f}"
    title_line += f"  [{merge_dir.name}]"
    draw.text((margin, 12), title_line, fill=(20, 20, 20), font=font_t)

    # legend
    legend_y = 34
    draw.rectangle([margin, legend_y, margin + 12, legend_y + 12], outline=COLOR_GT, width=2)
    draw.text((margin + 16, legend_y - 1), "GT", fill=COLOR_GT, font=font_xs)
    draw.rectangle([margin + 50, legend_y, margin + 62, legend_y + 12], outline=COLOR_NON_GT, width=2)
    draw.text((margin + 66, legend_y - 1), "non-GT", fill=COLOR_NON_GT, font=font_xs)
    draw.rectangle([margin + 130, legend_y, margin + 142, legend_y + 12], outline=COLOR_UNK, width=2)
    draw.text((margin + 146, legend_y - 1), "unknown", fill=COLOR_UNK, font=font_xs)

    y0 = title_h
    x = margin
    for i, col in enumerate(columns):
        info = col_infos[i]
        cw, ch = info["w"], body_h - 10
        # column outer: color by majority / any non-GT
        members = col["members"]
        if gt_set is None:
            bc = COLOR_UNK
        elif all(m in gt_set for m in members):
            bc = COLOR_GT
        elif any(m not in gt_set for m in members):
            bc = COLOR_NON_GT
        else:
            bc = COLOR_UNK
        if info["hi"]:
            bc = COLOR_HI
            draw.rectangle([x - 2, y0 - 2, x + cw + 1, y0 + ch + 1], outline=COLOR_HI, width=4)
            draw.text((x + 4, y0 - 16), "CHECK non-GT?  amplify", fill=COLOR_HI, font=font_b)
        else:
            draw.rectangle([x, y0, x + cw - 1, y0 + ch - 1], outline=bc, width=3)

        # header
        hdr = col["label"] if len(col["label"]) < 42 else ",".join(m.split("_")[-1] for m in members)
        draw.text((x + 4, y0 + 4), f"members: {', '.join(members)}", fill=(10, 10, 10), font=font_xs)

        my = y0 + label_h
        for mid, tid in enumerate(members):
            meta = track_meta(by_tid, tid)
            crops = load_crops(merge_dir, tid)
            three = _pick_three(crops)
            tw, th = info["tw"], info["th"]
            # member box
            mb_h = 3 * (th + 4) + meta_h + (12 if info["hi"] else 0)
            mb_c = border_color(tid, gt_set)
            if info["hi"] and tid == highlight_tid:
                mb_c = COLOR_HI
            draw.rectangle([x + 4, my, x + cw - 5, my + mb_h - 2], outline=mb_c, width=2)

            cx = x + (cw - tw) // 2
            cy = my + 4
            for cp in three:
                if cp is not None and cp.is_file():
                    img.paste(_thumb(cp, (tw, th)), (cx, cy))
                else:
                    draw.rectangle([cx, cy, cx + tw, cy + th], outline=(180, 180, 180))
                    draw.text((cx + 8, cy + th // 2 - 6), "no crop", fill=(160, 0, 0), font=font_xs)
                cy += th + 4

            ts = meta["t_start"]
            te = meta["t_end"]
            sim = meta["sim"]
            tspan = f"[{ts:.1f}–{te:.1f}]" if ts is not None else "[?–?]"
            sim_s = f"{sim:.3f}" if sim is not None else "?"
            ly = cy + 2
            draw.text((x + 8, ly), tid, fill=(0, 0, 0), font=font_s)
            draw.text((x + 8, ly + 14), f"{meta['cam']}  sim={sim_s}", fill=(40, 40, 40), font=font_xs)
            draw.text((x + 8, ly + 28), tspan, fill=(60, 60, 60), font=font_xs)
            my += mb_h + member_pad

        # arrow + edge info
        if i < len(columns) - 1:
            e = edges[i] if i < len(edges) else {}
            ax0 = x + cw
            ax1 = ax0 + gap_w
            mid_y = y0 + ch // 2
            draw.line([(ax0 + 4, mid_y), (ax1 - 12, mid_y)], fill=(0, 0, 0), width=2)
            draw.polygon(
                [(ax1 - 12, mid_y - 7), (ax1 - 3, mid_y), (ax1 - 12, mid_y + 7)],
                fill=(0, 0, 0),
            )
            hop = e.get("hop")
            dt = e.get("dt")
            sc = e.get("score")
            draw.text(
                (ax0 + 6, mid_y - 40),
                f"hop={hop}" if hop is not None else "hop=?",
                fill=(40, 40, 40),
                font=font_xs,
            )
            draw.text(
                (ax0 + 6, mid_y - 24),
                f"dt={dt:.1f}s" if dt is not None else "dt=?",
                fill=(40, 40, 40),
                font=font_xs,
            )
            if sc is not None:
                sc_c = (0, 120, 0) if sc >= 0 else (180, 0, 0)
                draw.text((ax0 + 6, mid_y + 10), f"edge={sc:+.2f}", fill=sc_c, font=font_s)
            elif e.get("missing"):
                draw.text((ax0 + 6, mid_y + 10), "no edge", fill=(180, 0, 0), font=font_xs)

        x += cw + gap_w

    foot = "  →  ".join(c["label"] for c in columns)
    draw.text((margin, title_h + body_h + 8), foot[:200], fill=(30, 30, 30), font=font_xs)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, quality=95)
    return out_png


def render_timeline(
    paths: list[dict],
    by_tid: dict,
    out_png: Path,
    *,
    title: str = "Top-3 timeline",
    gt_set: set[str] | None = None,
) -> Path:
    """x=time, y=camera; colored bars for tracks; line for path order."""
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]
    # collect cams & time range
    all_tids = []
    for p in paths:
        all_tids.extend(p.get("tids") or [])
    metas = [track_meta(by_tid, t) for t in all_tids if t in by_tid]
    if not metas:
        img = Image.new("RGB", (400, 120), BG)
        ImageDraw.Draw(img).text((20, 40), "no tracks", fill=(0, 0, 0), font=_font(14))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_png)
        return out_png

    cams = sorted({m["cam"] for m in metas})
    t_min = min(m["t_start"] for m in metas)
    t_max = max(m["t_end"] for m in metas)
    if t_max <= t_min:
        t_max = t_min + 1.0

    margin_l, margin_r, margin_t, margin_b = 90, 20, 50, 40
    row_h = 36
    width = 1100
    height = margin_t + margin_b + len(cams) * row_h + 30 * len(paths)
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    font = _font(12)
    font_xs = _font(10)
    draw.text((16, 12), title, fill=(20, 20, 20), font=_font(16))

    plot_w = width - margin_l - margin_r
    plot_h0 = margin_t

    def x_of(t: float) -> int:
        return margin_l + int((t - t_min) / (t_max - t_min) * plot_w)

    for yi, cam in enumerate(cams):
        y = plot_h0 + yi * row_h
        draw.text((8, y + 8), cam, fill=(30, 30, 30), font=font_xs)
        draw.line([(margin_l, y + row_h - 1), (width - margin_r, y + row_h - 1)], fill=(220, 220, 220))

    # draw each path's bars and connections
    for pi, p in enumerate(paths):
        color = palette[pi % len(palette)]
        cols = parse_super_columns(p)
        centers = []  # (x_mid, y_mid, tid)
        label = f"#{pi+1}"
        if p.get("path_probability") is not None:
            label += f" P={p['path_probability']:.3f}"
        draw.text((margin_l, height - margin_b + 4 + pi * 14), label, fill=color, font=font_xs)

        for col in cols:
            for tid in col["members"]:
                m = track_meta(by_tid, tid)
                if m["t_start"] is None:
                    continue
                yi = cams.index(m["cam"])
                y = plot_h0 + yi * row_h + 6 + pi * 3
                x0, x1 = x_of(m["t_start"]), x_of(max(m["t_end"], m["t_start"] + 0.3))
                if x1 <= x0:
                    x1 = x0 + 4
                # offset per path
                y1 = y + 10
                draw.rectangle([x0, y, x1, y1], fill=color + (0,), outline=color)
                # re-draw without alpha
                draw.rectangle([x0, y, x1, y1], outline=color, width=2)
                fill = (*color, )
                # solid fill
                for yy in range(y, y1 + 1):
                    draw.line([(x0, yy), (x1, yy)], fill=color)
                short = tid.split("_", 1)[-1]
                draw.text((x0 + 2, y - 1), short, fill=(255, 255, 255), font=font_xs)
                centers.append(((x0 + x1) // 2, (y + y1) // 2, tid))

        # connect in path order (first member of each column)
        pts = []
        for col in cols:
            tid = col["members"][0]
            m = track_meta(by_tid, tid)
            if m["t_start"] is None:
                continue
            yi = cams.index(m["cam"])
            y = plot_h0 + yi * row_h + 10 + pi * 3
            x_mid = x_of((m["t_start"] + m["t_end"]) / 2)
            pts.append((x_mid, y))
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=color, width=2)

    # time axis ticks
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = t_min + frac * (t_max - t_min)
        xx = x_of(t)
        draw.line([(xx, plot_h0 - 4), (xx, plot_h0 + len(cams) * row_h)], fill=(200, 200, 200))
        draw.text((xx - 10, plot_h0 + len(cams) * row_h + 2), f"{t:.0f}s", fill=(80, 80, 80), font=font_xs)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def build_gt_chain_path(by_tid: dict, calib: dict) -> dict:
    cols = [{"label": "{" + ",".join(m) + "}" if len(m) > 1 else m[0], "members": m} for m in EXPECTED_GT_SUPER]
    # fix label format to match style
    for c in cols:
        if len(c["members"]) > 1:
            c["label"] = "{" + ",".join(c["members"]) + "}"
    edges = reconstruct_edges_for_columns(cols, by_tid, calib)
    from path_enum_llr import SuperNode, path_score_llr

    sns = []
    for c in cols:
        tracks = [by_tid[m] for m in c["members"] if m in by_tid]
        emb = np.stack([t.emb for t in tracks]).mean(0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        sns.append(
            SuperNode(
                sid=c["label"],
                members=tracks,
                emb=emb,
                sim=float(np.mean([t.sim for t in tracks])),
                t_start=min(t.t_start for t in tracks),
                t_end=max(t.t_end for t in tracks),
                cams=sorted({t.cam for t in tracks}),
            )
        )
    score, node_ev = path_score_llr(sns, [e for e in edges if not e.get("missing")], calib)
    tids = []
    for c in cols:
        tids.extend(c["members"])
    return {
        "tids": tids,
        "super_labels": [c["label"] for c in cols],
        "edges": edges,
        "score": score,
        "node_evidence": node_ev,
        "path_probability": None,
        "note": "GT 11/11 expected super chain (display)",
    }


def ensure_crop_relinks(out_dir: Path, merge_dir: Path, tids: list[str]) -> None:
    """在 out_dir 建 crops/ 相對捷徑，供 HTML 引用。"""
    crops_root = out_dir / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    for tid in tids:
        cam, tid_s = tid.rsplit("_", 1)
        src_dir = OUTPUT_ROOT / f"{merge_dir.name}_{cam}"
        link = crops_root / cam
        if link.exists() or link.is_symlink():
            continue
        if src_dir.is_dir():
            try:
                link.symlink_to(src_dir.resolve())
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixed-dir",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "gt_calib_0507_fixed",
    )
    ap.add_argument(
        "--merge-dir",
        type=Path,
        default=QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "ground_truth_20260507.json",
    )
    ap.add_argument(
        "--calibration",
        type=Path,
        default=OUTPUT_ROOT / "path_enum_llr" / "calibration_gt0507.pkl",
    )
    args = ap.parse_args()

    fixed_dir = args.fixed_dir.resolve()
    merge_dir = args.merge_dir.resolve()
    dataset = merge_dir.name
    gt_set = load_gt_set(args.gt.resolve())
    calib = llr.load_calibration(args.calibration.resolve())

    top_json = fixed_dir / f"{dataset}_llr_top1.json"
    summary = json.loads(top_json.read_text(encoding="utf-8"))

    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(merge_dir))
    # 與修正後一致的門檻，僅為取 crop／track meta
    import llr_gate_config as gates

    gates.apply_llr_emb_gates(True)
    tracks = pes.load_tracks(str(merge_dir))
    by_tid = {t.tid: t for t in tracks}

    # 用同一設定重跑拿齊 Top-10 完整 edges（不改公式）；若 top1 tids 與 JSON 一致則採用
    print("enumerate for viz edges…")
    (
        _tr,
        scored,
        maximal,
        _ne,
        _nodes,
        _srep,
        _gate,
        _opt,
    ) = llr.run_llr(
        merge_dir,
        calib,
        use_emb_gate_fix=True,
        use_supernode=True,
        use_node_evidence=True,
    )

    # 以 fixed JSON 的 top1 / top10 順序為準；edges 從同 tids 的 maximal 補齊，否則重算
    fixed_top1 = summary["top1"]
    fixed_top10 = summary.get("top10_paths") or []
    maximal_by_tids = {tuple(p["tids"]): p for p in maximal}

    def resolve_path(spec: dict) -> dict:
        tids = spec.get("tids") or []
        key = tuple(tids)
        if key in maximal_by_tids:
            full = dict(maximal_by_tids[key])
            # 保留 JSON 的分數／機率（以 fixed 為準）
            if "score" in spec:
                full["score"] = spec["score"]
            if "path_probability" in spec:
                full["path_probability"] = spec["path_probability"]
            if spec.get("super_labels"):
                full["super_labels"] = spec["super_labels"]
            return full
        # 無完整 edges：用 JSON + 重算邊
        cols_labels = spec.get("super_labels")
        path = {
            "tids": tids,
            "super_labels": cols_labels,
            "score": spec.get("score"),
            "path_probability": spec.get("path_probability"),
            "edges": spec.get("edges") or [],
        }
        if not path["edges"] and cols_labels:
            cols = parse_super_columns(path)
            path["edges"] = reconstruct_edges_for_columns(cols, by_tid, calib)
        elif path.get("edges") and path["edges"][0].get("from_members") is None:
            cols = parse_super_columns(path)
            path["edges"] = reconstruct_edges_for_columns(cols, by_tid, calib)
        return path

    # 以目前修正後管線 rematch 的 Top-k 為準（與 comparison 報告一致），
    # 並回寫 fixed 目錄 JSON，避免舊檔未含 {23_8,22_19} 合併。
    viz_paths = maximal[:10]
    if not viz_paths:
        viz_paths = [fixed_top1]

    # 回寫 top1 JSON（輸出同步，不改計分公式）
    alt = llr.best_disjoint_alternative(maximal)
    refreshed = llr.build_summary_json(
        merge_dir,
        scored,
        maximal,
        None,
        alt,
        _ne,
        len(tracks),
        super_report=_srep,
        gate_info=_gate,
        options=_opt,
    )
    top_json.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"refreshed {top_json}")

    print(f"Top-1 path: {' -> '.join(viz_paths[0].get('super_labels') or viz_paths[0]['tids'])}")

    out_files = []

    # --- 輸出一：Top-1 ---
    p1 = fixed_dir / f"{dataset}_top1_collage.png"
    render_path_collage(
        merge_dir,
        viz_paths[0],
        by_tid,
        p1,
        title="Top-1 (fixed)",
        gt_set=gt_set,
        highlight_tid="K8-09_42",
        calib=calib,
    )
    out_files.append(p1)
    print(f"wrote {p1}")

    # --- 輸出二：Top-2 / Top-3 + timeline ---
    collage_map = {1: p1.name}
    for rank in (2, 3):
        if rank - 1 < len(viz_paths):
            outp = fixed_dir / f"{dataset}_top{rank}_collage.png"
            render_path_collage(
                merge_dir,
                viz_paths[rank - 1],
                by_tid,
                outp,
                title=f"Top-{rank} (fixed)",
                gt_set=gt_set,
                highlight_tid="K8-09_42",
                calib=calib,
            )
            out_files.append(outp)
            collage_map[rank] = outp.name
            print(f"wrote {outp}")

    tl = fixed_dir / f"{dataset}_top3_timeline.png"
    render_timeline(
        viz_paths[:3],
        by_tid,
        tl,
        title=f"{dataset} Top-3 timeline",
        gt_set=gt_set,
    )
    out_files.append(tl)
    print(f"wrote {tl}")

    # --- 輸出三：GT 全鏈 ---
    gt_path = build_gt_chain_path(by_tid, calib)
    gt_png = fixed_dir / f"{dataset}_gt_chain_collage.png"
    render_path_collage(
        merge_dir,
        gt_path,
        by_tid,
        gt_png,
        title="GT 11/11 chain (expected supers)",
        gt_set=gt_set,
        highlight_tid=None,
        calib=calib,
    )
    out_files.append(gt_png)
    print(f"wrote {gt_png}")

    # --- 輸出四：HTML ---
    # 為 HTML 建 crop 相對連結：把用到的 crop 以 crops/cam/name 可達
    all_tids = []
    for p in viz_paths[:10]:
        all_tids.extend(p.get("tids") or [])
    all_tids.extend(gt_path["tids"])
    ensure_crop_relinks(fixed_dir, merge_dir, sorted(set(all_tids)))

    # 改 HTML 用相對路徑：crops/{cam}/{filename}
    def write_html_v2():
        rows = []
        for i, p in enumerate(viz_paths[:10], 1):
            cols = parse_super_columns(p)
            edges = p.get("edges") or []
            if len(edges) != max(0, len(cols) - 1):
                edges = reconstruct_edges_for_columns(cols, by_tid, calib)
            thumbs = []
            for col in cols:
                tid = col["members"][0]
                cam = tid.rsplit("_", 1)[0]
                crops = load_crops(merge_dir, tid)
                three = _pick_three(crops)
                mid = three[1]
                border = "#228B22" if tid in gt_set else "#C82828"
                if mid is not None and mid.is_file():
                    src = f"crops/{cam}/{mid.name}"
                    thumbs.append(
                        f'<div style="display:inline-block;margin:2px;border:3px solid {border};padding:2px;vertical-align:top">'
                        f'<img src="{html.escape(src)}" height="80"/>'
                        f'<div style="font:11px sans-serif;max-width:100px">{html.escape(col["label"])}</div></div>'
                    )
                else:
                    thumbs.append(
                        f'<div style="display:inline-block;border:2px solid {border};padding:4px">{html.escape(col["label"])}</div>'
                    )

            edge_rows = []
            for e in edges:
                dt_s = f"{e['dt']:.2f}" if e.get("dt") is not None else "?"
                emb_s = f"{e['emb']:.3f}" if e.get("emb") is not None else "?"
                lemb = e.get("LLR_emb")
                lemb_s = f"{lemb:+.3f}" if lemb is not None else "?"
                sc = e.get("score")
                sc_s = f"{sc:+.3f}" if sc is not None else "?"
                edge_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(e.get('from_super', e.get('from'))))}</td>"
                    f"<td>{html.escape(str(e.get('to_super', e.get('to'))))}</td>"
                    f"<td>{html.escape(str(e.get('hop')))}</td>"
                    f"<td>{html.escape(dt_s)}</td>"
                    f"<td>{html.escape(emb_s)}</td>"
                    f"<td>{html.escape(str(e.get('LLR_dt')))}</td>"
                    f"<td>{html.escape(lemb_s)}</td>"
                    f"<td>{html.escape(str(e.get('LLR_dH')))}</td>"
                    f"<td>{html.escape(sc_s)}</td>"
                    "</tr>"
                )
            link = collage_map.get(i, "")
            a = f'<a href="{html.escape(link)}">collage</a>' if link else ""
            P = p.get("path_probability")
            P_s = f"{P:.4f}" if P is not None else "—"
            rows.append(
                f"<details {'open' if i==1 else ''}>"
                f"<summary><b>#{i}</b> score={p.get('score', float('nan')):.3f} P={P_s} {a}<br/>{''.join(thumbs)}</summary>"
                "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font:12px sans-serif;margin:8px 0'>"
                "<tr><th>from</th><th>to</th><th>hop</th><th>dt</th><th>emb</th><th>LLR_dt</th><th>LLR_emb</th><th>LLR_dH</th><th>edge</th></tr>"
                + "".join(edge_rows)
                + "</table>"
                f"<div style='font:12px monospace'>{html.escape(' → '.join(p.get('super_labels') or p.get('tids') or []))}</div>"
                "</details>"
            )

        gt_cols = parse_super_columns(gt_path)
        gt_thumbs = []
        for col in gt_cols:
            tid = col["members"][0]
            cam = tid.rsplit("_", 1)[0]
            crops = load_crops(merge_dir, tid)
            mid = _pick_three(crops)[1]
            if mid is not None and mid.is_file():
                gt_thumbs.append(
                    f'<img src="crops/{cam}/{mid.name}" height="70" style="border:3px solid #228B22;margin:2px" title="{html.escape(col["label"])}"/>'
                )
        body = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<title>{html.escape(dataset)} paths</title>
<style>
body{{font-family:sans-serif;margin:16px;background:#f7f7f7}}
summary{{cursor:pointer;padding:8px;background:#fff;border:1px solid #ddd}}
details{{margin-bottom:10px}}
</style></head><body>
<h1>{html.escape(dataset)} — 修正後路徑總覽</h1>
<p>框線：綠=GT、紅=非GT（僅檢視）。GT 未進入演算法。</p>
<h2>GT 11/11 chain</h2>
<p><a href="{dataset}_gt_chain_collage.png">gt_chain_collage</a> {' '.join(gt_thumbs)}</p>
<h2>Top-10</h2>
{''.join(rows)}
</body></html>"""
        out_html = fixed_dir / f"{dataset}_paths.html"
        out_html.write_text(body, encoding="utf-8")
        return out_html

    html_path = write_html_v2()
    out_files.append(html_path)
    print(f"wrote {html_path}")

    # also save gt chain json snippet for reference
    (fixed_dir / f"{dataset}_gt_chain.json").write_text(
        json.dumps(gt_path, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== 輸出檔案 ===")
    for f in out_files:
        print(f.resolve())


if __name__ == "__main__":
    main()
