# -*- coding: utf-8 -*-
"""
拓撲表倒出 + 0528 Top-1 依序時間圖（不改演算法）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import visualize_fixed_paths as viz  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

OUT_DIR = OUTPUT_ROOT / "path_enum_llr" / "viz_0528"
MERGE = QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528"
TOP1_JSON = OUTPUT_ROOT / "path_enum_llr" / "人員追蹤_20260528_llr_top1.json"
GT_PATH = OUTPUT_ROOT / "path_enum_llr" / "ground_truth_20260528.json"
DUMP_PATH = OUTPUT_ROOT / "path_enum_llr" / "topology_dump.txt"

# 使用者點名的非 GT（檢視用紅框）
NON_GT_EMPHASIS = {"K8-07_1", "K8-07_93", "K8-09_96", "K8-07_139", "K8-09_167"}


def has_h(cam_a: str, cam_b: str) -> bool:
    return (cam_a, cam_b) in pes.H_MATRICES or (cam_b, cam_a) in pes.H_MATRICES


def h_dirs(cam_a: str, cam_b: str) -> list[str]:
    out = []
    if (cam_a, cam_b) in pes.H_MATRICES:
        out.append(f"{cam_a}->{cam_b}")
    if (cam_b, cam_a) in pes.H_MATRICES:
        out.append(f"{cam_b}->{cam_a}")
    return out


def dump_topology(top1: dict) -> str:
    lines = []
    lines.append("=== 拓撲表倒出（PERSON 模式）===")
    lines.append(f"來源：path_enum_scoring.PERSON_* ；HOMOGRAPHY_DIR={pes.HOMOGRAPHY_DIR}")
    lines.append(f"MIN_TRANSIT hop1={pes.DEFAULT_MIN_TRANSIT_HOP1} hop2={pes.DEFAULT_MIN_TRANSIT_HOP2}")
    lines.append(f"DT_MAX={pes.DT_MAX}  TOL(非OVERLAP)={pes.TOL}")
    lines.append("")

    adj = sorted(tuple(sorted(p)) for p in pes.PERSON_ADJACENT)
    lines.append(f"--- PERSON_ADJACENT（共 {len(adj)} 對）---")
    for a, b in adj:
        hs = h_dirs(a, b)
        h_s = ", ".join(hs) if hs else "無 H"
        ov = pes.PERSON_OVERLAP_PAIRS.get((a, b))
        ov_s = f"OVERLAP tol={ov}s" if ov is not None else "非 OVERLAP"
        lines.append(f"  {a} ↔ {b}   | {ov_s} | H: {h_s}")
    lines.append("")

    lines.append(f"--- PERSON_OVERLAP_PAIRS（共 {len(pes.PERSON_OVERLAP_PAIRS)} 對）---")
    for (a, b), tol in sorted(pes.PERSON_OVERLAP_PAIRS.items()):
        hs = h_dirs(a, b)
        h_s = ", ".join(hs) if hs else "無 H"
        lines.append(f"  {a} ↔ {b}   tol={tol}s   H: {h_s}")
    lines.append("")

    # 所有 ADJACENT 對的 H 一覽（含僅 H、無 ADJ 的？僅列 ADJ+OVERLAP 已覆蓋；另列所有已載入 H）
    lines.append(f"--- 已載入 Homography 方向（共 {len(pes.H_MATRICES)}）---")
    for a, b in sorted(pes.H_MATRICES.keys()):
        key = tuple(sorted((a, b)))
        in_adj = key in pes.PERSON_ADJACENT or key in {tuple(sorted(p)) for p in pes.PERSON_ADJACENT}
        in_ov = key in pes.PERSON_OVERLAP_PAIRS
        flags = []
        if in_adj:
            flags.append("ADJACENT")
        if in_ov:
            flags.append(f"OVERLAP tol={pes.PERSON_OVERLAP_PAIRS[key]}s")
        if not flags:
            flags.append("（非人員 ADJ/OVERLAP）")
        lines.append(f"  {a}->{b}   {' | '.join(flags)}")
    lines.append("")

    # Top-1 edges
    edges = top1.get("edges") or []
    lines.append("--- 0528 Top-1 實際邊（鏡頭對 + dt + hop）---")
    lines.append(f"路徑：{' -> '.join(top1.get('super_labels') or [])}")
    lines.append("")
    short_dt_non_ov = []
    for i, e in enumerate(edges, 1):
        fa = (e.get("from_members") or [e["from"]])[0]
        ta = (e.get("to_members") or [e["to"]])[0]
        cam_f = fa.rsplit("_", 1)[0]
        cam_t = ta.rsplit("_", 1)[0]
        # 更準：用 from_super / to_super 的 cams
        from_mems = e.get("from_members") or [e["from"]]
        to_mems = e.get("to_members") or [e["to"]]
        cams_f = sorted({m.rsplit("_", 1)[0] for m in from_mems})
        cams_t = sorted({m.rsplit("_", 1)[0] for m in to_mems})
        cam_pair = f"{'/'.join(cams_f)} → {'/'.join(cams_t)}"
        key_pairs = []
        for cf in cams_f:
            for ct in cams_t:
                key_pairs.append(tuple(sorted((cf, ct))))
        is_ov = any(k in pes.PERSON_OVERLAP_PAIRS for k in key_pairs)
        is_adj = any(k in set(tuple(sorted(p)) for p in pes.PERSON_ADJACENT) for k in key_pairs)
        has_any_h = any(has_h(cf, ct) for cf in cams_f for ct in cams_t)
        dt = float(e.get("dt") or 0.0)
        hop = e.get("hop")
        via = f"{e.get('from')}->{e.get('to')}"
        flag = ""
        if dt < 3.0 and not is_ov:
            flag = "  ★ dt<3s 且非 OVERLAP"
            short_dt_non_ov.append((i, e, cam_pair, dt, hop, is_adj, has_any_h, via))
        ov_tag = "OVERLAP" if is_ov else ("ADJACENT" if is_adj else "其他")
        h_tag = "有H" if has_any_h else "無H"
        lines.append(
            f"  [{i:02d}] {e.get('from_super')} → {e.get('to_super')}   "
            f"鏡頭 {cam_pair}   dt={dt:.2f}s  hop={hop}  {ov_tag}/{h_tag}  "
            f"via={via}{flag}"
        )

    lines.append("")
    lines.append(f"--- 特別標注：dt < 3s 且非 OVERLAP 的邊（共 {len(short_dt_non_ov)}）---")
    lines.append("（供人工核對「相鄰但有間隔／邊界相接」的配對）")
    if not short_dt_non_ov:
        lines.append("  （無）")
    for i, e, cam_pair, dt, hop, is_adj, has_any_h, via in short_dt_non_ov:
        lines.append(
            f"  [{i:02d}] {e.get('from_super')} → {e.get('to_super')}  "
            f"{cam_pair}  dt={dt:.2f}s hop={hop}  "
            f"{'ADJACENT' if is_adj else '非ADJ'}  {'有H' if has_any_h else '無H'}  via={via}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _thumb(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None or not path.is_file():
        im = Image.new("RGB", size, (230, 230, 230))
        return im
    return viz._thumb(path, size)


def render_top1_sequence(
    top1: dict,
    by_tid: dict,
    merge_dir: Path,
    gt_set: set[str],
    out_png: Path,
    segments: list | None = None,
    title: str | None = None,
) -> tuple[Path, list[str]]:
    """
    橫軸=絕對時間；每一路徑節點佔一列（依路徑順序由上而下）；
    超節點成員在該列內上下堆疊；空檔在橫軸標 dt/hop；
    GT=綠、非GT=紅。
    """
    columns = viz.parse_super_columns(top1)
    edges = top1.get("edges") or []

    # 附加 segment（若有 seg2+）
    blocks: list[tuple[str, list[dict], list[dict], float | None]] = [
        ("seg1", columns, edges, None)
    ]
    for seg in segments or []:
        if int(seg.get("segment") or 1) <= 1:
            continue
        fake = {
            "super_labels": seg.get("super_labels"),
            "tids": seg.get("tids"),
            "edges": seg.get("edges") or [],
        }
        cols = viz.parse_super_columns(fake)
        blocks.append(
            (
                f"seg{seg['segment']}",
                cols,
                fake.get("edges") or [],
                float(seg.get("gap_after_prev_sec") or 0.0),
            )
        )

    all_members: list[str] = []
    for _, cols, _, _ in blocks:
        for c in cols:
            all_members.extend(c["members"])
    metas = [viz.track_meta(by_tid, t) for t in all_members if t in by_tid]
    t_min = min(m["t_start"] for m in metas)
    t_max = max(m["t_end"] for m in metas)
    if t_max <= t_min:
        t_max = t_min + 1.0

    thumb_w, thumb_h = 64, 86
    mem_h = thumb_h + 36
    lane_pad = 18
    title_h = 70
    axis_h = 50
    margin_l, margin_r, margin_b = 160, 30, 50

    # 每欄列高 = 成員數 * mem_h
    lane_heights = []
    for _, cols, _, gap in blocks:
        if gap is not None:
            lane_heights.append(36)  # 空窗列
        for c in cols:
            lane_heights.append(max(1, len(c["members"])) * mem_h + lane_pad)

    body_h = sum(lane_heights) + 20
    px_per_sec = 2.4
    plot_w = max(1400, int((t_max - t_min) * px_per_sec) + 180)
    width = margin_l + plot_w + margin_r
    height = title_h + body_h + axis_h + margin_b

    img = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    font_t = viz._font(16)
    font_s = viz._font(11)
    font_xs = viz._font(9)
    font_b = viz._font(11, bold=True)

    score = top1.get("score")
    pprob = top1.get("path_probability")
    n_seg = int(top1.get("n_segments") or (1 + sum(1 for s in (segments or []) if int(s.get("segment") or 1) > 1)))
    default_title = (
        f"Top-1 sequence  n_seg={n_seg}  score={score:.3f}  P={pprob:.4f}"
        if score is not None
        else "Top-1 sequence"
    )
    draw.text(
        (16, 8),
        title or default_title,
        fill=(20, 20, 20),
        font=font_t,
    )
    ly = 36
    draw.rectangle([16, ly, 30, ly + 14], outline=(34, 139, 34), width=3)
    draw.text((34, ly), "GT", fill=(34, 139, 34), font=font_xs)
    draw.rectangle([70, ly, 84, ly + 14], outline=(200, 40, 40), width=3)
    draw.text((88, ly), "non-GT（07_1/07_93/09_96/07_139/09_167 等）", fill=(200, 40, 40), font=font_xs)
    draw.text((420, ly), "列=路徑順序；橫軸=絕對時間；超節點成員同列堆疊", fill=(80, 80, 80), font=font_xs)

    def x_of(t: float) -> int:
        return margin_l + int((t - t_min) / (t_max - t_min) * plot_w)

    axis_y = title_h + body_h + 4
    draw.line([(margin_l, axis_y), (margin_l + plot_w, axis_y)], fill=(50, 50, 50), width=2)
    for i in range(0, int(t_max) + 1, 50):
        if i < t_min - 1:
            continue
        xx = x_of(float(i))
        if xx < margin_l or xx > margin_l + plot_w:
            continue
        draw.line([(xx, axis_y - 5), (xx, axis_y + 5)], fill=(50, 50, 50))
        draw.text((xx - 10, axis_y + 8), f"{i}s", fill=(70, 70, 70), font=font_xs)

    crop_log: list[str] = []
    y = title_h
    prev_box_x1 = None

    for bi, (bname, cols, bedges, gap) in enumerate(blocks):
        if gap is not None:
            # 空窗列
            draw.rectangle(
                [margin_l, y, margin_l + plot_w, y + 28],
                fill=(255, 245, 230),
                outline=(200, 120, 40),
            )
            draw.text(
                (margin_l + 8, y + 6),
                f"觀測空窗 {gap:.1f}s  →  {bname}",
                fill=(160, 80, 0),
                font=font_b,
            )
            y += 36

        for ci, col in enumerate(cols):
            members = col["members"]
            spans = []
            for tid in members:
                m = viz.track_meta(by_tid, tid)
                if m["t_start"] is None:
                    continue
                spans.append((float(m["t_start"]), float(m["t_end"]), tid, m))
            spans.sort(key=lambda z: (z[0], z[2]))
            n = max(1, len(spans))
            lane_h = n * mem_h + lane_pad
            # 左側標籤
            draw.text((8, y + 8), col["label"][:28], fill=(30, 30, 30), font=font_xs)
            if bi == 0 and ci < len(edges):
                # path order index
                draw.text((8, y + 22), f"#{ci+1}", fill=(100, 100, 100), font=font_xs)

            if not spans:
                y += lane_h
                continue

            t0 = min(s[0] for s in spans)
            t1 = max(s[1] for s in spans)
            x0 = x_of(t0)
            x1 = max(x_of(max(t1, t0 + 0.3)), x0 + thumb_w + 160)

            if all(tid in gt_set for tid in members):
                bc = (34, 139, 34)
            else:
                bc = (200, 40, 40)

            # 外框（超節點合併）
            pad = 3
            draw.rectangle(
                [x0 - pad, y + 2, x1 + pad, y + n * mem_h + 4],
                outline=bc,
                width=3 if len(members) > 1 else 2,
            )
            if len(members) > 1:
                draw.text((x0, y - 1), "超節點合併", fill=bc, font=font_xs)

            for mi, (ts, te, tid, m) in enumerate(spans):
                yy = y + mi * mem_h + 4
                bx0 = x_of(ts)
                bx1 = max(x_of(max(te, ts + 0.25)), bx0 + 4)
                # 時間橫條（細）
                bar_y0, bar_y1 = yy + 2, yy + 14
                draw.rectangle([bx0, bar_y0, bx1, bar_y1], fill=bc, outline=bc)
                # crop
                crops = viz.load_crops(merge_dir, tid)
                mid = viz._pick_three(crops)[1]
                thumb = _thumb(mid, (thumb_w, thumb_h))
                cx = min(max(bx0, x0), max(x0, x1 - thumb_w))
                img.paste(thumb, (cx, yy + 16))
                draw.rectangle(
                    [cx, yy + 16, cx + thumb_w, yy + 16 + thumb_h],
                    outline=bc,
                    width=2,
                )
                sim_s = f"{m['sim']:.3f}" if m["sim"] is not None else "?"
                draw.text(
                    (cx + thumb_w + 4, yy + 18),
                    f"{tid}  {m['cam']}",
                    fill=(20, 20, 20),
                    font=font_xs,
                )
                draw.text(
                    (cx + thumb_w + 4, yy + 32),
                    f"sim={sim_s}  [{ts:.1f},{te:.1f}]",
                    fill=(60, 60, 60),
                    font=font_xs,
                )
                if mid is not None:
                    crop_log.append(f"{tid}\t{mid}")
                else:
                    crop_log.append(f"{tid}\t(no crop)")

            # 與前一欄的邊：標在列左側（路徑序），避免絕對時間與路徑序不一致時軸上錯位
            if ci > 0 and ci - 1 < len(bedges):
                e = bedges[ci - 1]
                dt = e.get("dt")
                hop = e.get("hop")
                label = f"↑ dt={dt:.1f}s hop={hop}" if dt is not None else f"↑ hop={hop}"
                draw.text((8, y + lane_h - 16), label, fill=(40, 40, 140), font=font_xs)
                # 若時間上嚴格先後，另在橫軸畫區間
                if prev_box_x1 is not None and x0 > prev_box_x1 + 4:
                    ay = axis_y - 18
                    draw.line([(prev_box_x1, ay), (x0 - pad, ay)], fill=(80, 80, 160), width=2)
                    draw.text(
                        ((prev_box_x1 + x0) // 2 - 28, ay - 14),
                        f"dt={dt:.1f}s",
                        fill=(40, 40, 140),
                        font=font_xs,
                    )

            prev_box_x1 = x1 + pad
            draw.line(
                [(margin_l, y + n * mem_h + lane_pad - 4), (margin_l + plot_w, y + n * mem_h + lane_pad - 4)],
                fill=(230, 230, 230),
            )
            y += lane_h

        prev_box_x1 = None  # 下一段重新起算橫軸標註

    draw.text(
        (16, height - 32),
        "紅=非GT／綠=GT。僅供人工檢視，不改演算法。",
        fill=(70, 70, 70),
        font=font_xs,
    )
    draw.text(
        (16, height - 18),
        f"時間範圍 [{t_min:.1f}, {t_max:.1f}]s",
        fill=(90, 90, 90),
        font=font_xs,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png, crop_log


def main():
    pes.SIM_MIN = 0.85
    pes.configure_for_input(str(MERGE))
    pes._load_h_matrices()

    data = json.loads(TOP1_JSON.read_text(encoding="utf-8"))
    top1 = data["top1"]
    segments = data.get("segments") or []
    gt_set = set(json.loads(GT_PATH.read_text(encoding="utf-8"))["person_tids"])

    # 1) topology dump
    dump = dump_topology(top1)
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(dump, encoding="utf-8")
    print(f"拓撲倒出：{DUMP_PATH}")
    print(dump)

    # 2) sequence viz — 需 track meta + emb 僅為 crop；用 load_tracks
    tracks = pes.load_tracks(str(MERGE))
    by_tid = {t.tid: t for t in tracks}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "人員追蹤_20260528_top1_sequence.png"
    png, crop_log = render_top1_sequence(
        top1, by_tid, MERGE, gt_set, out_png, segments=segments
    )
    crop_txt = OUT_DIR / "人員追蹤_20260528_top1_crop_list.txt"
    crop_txt.write_text(
        "tid\tcrop_path\n" + "\n".join(crop_log) + "\n", encoding="utf-8"
    )
    print(f"時間圖：{png}")
    print(f"crop 清單：{crop_txt}")
    print(f"segments in JSON: {len(segments)}")


if __name__ == "__main__":
    main()
