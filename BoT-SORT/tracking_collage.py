#!/usr/bin/env python3
"""BoT-SORT 追蹤結果拼圖（PIL）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

TILE_W = 120
TILE_H = 200
CAPTION_H = 40
CARD_PAD = 4
HEADER_H = 24
ROW_SEP = 3
ROW_GAP = 10
REP_GAP = 28
TILE_GAP = 8
OUTER_PAD = 16
MIN_CANVAS_W = 2000
BG = (245, 245, 245)
REP_OUTLINE = (0, 160, 0)
TILE_OUTLINE = (180, 180, 180)


def _font(size: int = 12):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_tile(path: str, tw: int = TILE_W, th: int = TILE_H) -> Image.Image:
    p = Path(path)
    if not p.is_file():
        alt = Path(str(path).replace("/home/M133040024/ASE/", "/mnt/10THDD/M133040024/SSD/ASE/"))
        if alt.is_file():
            p = alt
    try:
        return ImageOps.pad(Image.open(p).convert("RGB"), (tw, th), color=(0, 0, 0))
    except Exception:
        img = Image.new("RGB", (tw, th), color=(30, 30, 30))
        ImageDraw.Draw(img).text((6, 6), "load fail", fill=(220, 60, 60))
        return img


def _card_size() -> Tuple[int, int]:
    return TILE_W + CARD_PAD * 2, TILE_H + CAPTION_H + CARD_PAD * 2


def _draw_tile_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    rec: Dict[str, Any],
    outline: Tuple[int, int, int],
    width: int,
    font,
) -> None:
    cw, ch = _card_size()
    draw.rectangle([(x, y), (x + cw - 1, y + ch - 1)], outline=outline, width=width)
    ix = x + CARD_PAD
    iy = y + CARD_PAD
    canvas.paste(_load_tile(rec["crop_path"]), (ix, iy))
    name = Path(rec["crop_path"]).name
    line1 = name if len(name) <= 28 else "..." + name[-25:]
    line2 = f"f={rec['frame']}  s={rec['score']:.2f}"
    ty = iy + TILE_H + 4
    draw.text((ix, ty), line1, fill=(0, 0, 0), font=font)
    draw.text((ix, ty + 14), line2, fill=(60, 60, 60), font=font)


def _measure_tiles_block(rest: List[Dict[str, Any]], tiles_area_w: int) -> int:
    if not rest:
        return _card_size()[1]
    cw, ch = _card_size()
    step = cw + TILE_GAP
    per_row = max(1, (tiles_area_w + TILE_GAP) // step)
    n_rows = (len(rest) + per_row - 1) // per_row
    return n_rows * ch + (n_rows - 1) * TILE_GAP


def save_tracking_collage(
    *,
    track_rows: List[Tuple[int, List[Dict[str, Any]]]],
    out_path: Path,
    canvas_w: int = MIN_CANVAS_W,
) -> None:
    if not track_rows:
        print("[WARN] 無 track 可拼圖")
        return

    font = _font(11)
    header_font = _font(12)
    card_w, card_h = _card_size()

    # 固定最小寬度；僅在代表圖+至少一張圖卡無法容納時略為加寬
    min_row_w = OUTER_PAD + card_w + REP_GAP + card_w + OUTER_PAD
    canvas_w = max(canvas_w, MIN_CANVAS_W, min_row_w)
    tiles_area_w = canvas_w - 2 * OUTER_PAD - card_w - REP_GAP
    right_limit = canvas_w - OUTER_PAD

    blocks: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]], int]] = []
    total_h = OUTER_PAD
    for tid, crops in track_rows:
        crops_sorted = sorted(crops, key=lambda r: (r["frame"], -r["score"]))
        rep = max(crops_sorted, key=lambda r: r["score"])
        rest = [c for c in crops_sorted if c is not rep]
        t0 = crops_sorted[0]["absolute_timestamp"]
        t1 = crops_sorted[-1]["absolute_timestamp"]
        header = f"track_id={tid}  n={len(crops)}  {t0} --> {t1}"
        tiles_h = _measure_tiles_block(rest, tiles_area_w)
        cards_h = max(card_h, tiles_h)
        block_h = HEADER_H + ROW_GAP + cards_h
        blocks.append((header, rep, rest, block_h))
        total_h += block_h + ROW_SEP + ROW_GAP

    canvas_h = total_h + OUTER_PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=BG)
    draw = ImageDraw.Draw(canvas)
    y = OUTER_PAD

    for header, rep, rest, block_h in blocks:
        draw.rectangle(
            [(OUTER_PAD, y), (canvas_w - OUTER_PAD, y + HEADER_H - 1)],
            fill=(255, 255, 255),
            outline=(200, 200, 200),
        )
        draw.text((OUTER_PAD + 6, y + 5), header, fill=(0, 0, 0), font=header_font)
        y_cards = y + HEADER_H + ROW_GAP

        _draw_tile_card(
            canvas, draw, x=OUTER_PAD, y=y_cards, rec=rep,
            outline=REP_OUTLINE, width=3, font=font,
        )

        x0 = OUTER_PAD + card_w + REP_GAP
        cx, cy = x0, y_cards
        for rec in rest:
            if cx + card_w > right_limit and cx > x0:
                cx = x0
                cy += card_h + TILE_GAP
            _draw_tile_card(
                canvas, draw, x=cx, y=cy, rec=rec,
                outline=TILE_OUTLINE, width=1, font=font,
            )
            cx += card_w + TILE_GAP

        y += block_h + ROW_SEP + ROW_GAP
        sep_y = y - ROW_GAP - ROW_SEP // 2
        draw.line(
            [(OUTER_PAD, sep_y), (canvas_w - OUTER_PAD, sep_y)],
            fill=(210, 210, 210),
            width=ROW_SEP,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"[OK] 拼圖 -> {out_path} ({canvas_w}x{canvas_h}, {len(track_rows)} tracks)")
