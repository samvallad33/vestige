#!/usr/bin/env python3
"""Render the Silent Rotation launch figure from verified EVIDENCE.md counts."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_HERO = ROOT / "silent-rotation-hero.png"
OUT_OG = ROOT / "silent-rotation-og.png"

VOID = (2, 4, 10, 255)
CARD = (8, 14, 26, 255)
LINE = (34, 199, 222, 42)
CYAN = (34, 199, 222, 255)
MINT = (125, 255, 176, 255)
WRONG = (255, 92, 110, 255)
ORANGE = (255, 158, 92, 255)
MUTED = (146, 165, 180, 255)
TEXT = (237, 250, 255, 255)
DIM = (92, 110, 124, 255)
HN = "/Library/Fonts/HelveticaNeue.ttc"
MENLO = "/Library/Fonts/Menlo.ttc"


def font(path, size, index=0):
    return ImageFont.truetype(path, size=size, index=index)


def rounded(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw, xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=fnt, fill=fill)


def glow(base, cx, cy, rx, ry, color, blur=48):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))


def pct_label(hit, n):
    pct = 100.0 * hit / n
    if abs(pct - round(pct)) < 1e-9:
        return f"{hit}/{n}   {pct:.0f}%"
    return f"{hit}/{n}   {pct:.1f}%"


def draw_card(draw, x, y, w, h, name, wrong, total, correct, split, accent, hero, fonts):
    outline = accent if hero else (255, 255, 255, 28)
    rounded(draw, (x, y, x + w, y + h), max(16, int(h * 0.06)), CARD, outline=outline, width=2 if hero else 1)
    cx = x + w / 2
    text_center(draw, (cx, y + h * 0.10), name, fonts["name"], MUTED)
    text_center(draw, (cx, y + h * 0.18), "CONVERGED ON THE WRONG KEY", fonts["small"], DIM)
    text_center(draw, (cx, y + h * 0.38), f"{wrong}", fonts["num"], accent)
    text_center(draw, (cx, y + h * 0.54), f"of {total} trials", fonts["frac"], MUTED)
    rate = 100.0 * wrong / total
    text_center(draw, (cx, y + h * 0.64), f"{rate:.0f}% shipped the decoy", fonts["cap"], TEXT if hero else MUTED)
    bar_x, bar_y, bar_w, bar_h = x + w * 0.08, y + h * 0.74, w * 0.84, max(6, h * 0.025)
    rounded(draw, (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), 5, (255, 255, 255, 18))
    fill_w = int(bar_w * wrong / total) if wrong else 0
    if fill_w:
        rounded(draw, (bar_x, bar_y, bar_x + max(fill_w, bar_h), bar_y + bar_h), 5, accent)
    text_center(draw, (cx, y + h * 0.88), f"correct {correct}   split {split}", fonts["small"], DIM)


def render(width, height, og=False):
    img = Image.new("RGBA", (width, height), VOID)
    glow(img, width * 0.72, height * 0.42, int(width * 0.18), int(height * 0.2), (34, 199, 222, 38), blur=64)
    glow(img, width * 0.22, height * 0.78, int(width * 0.16), int(height * 0.14), (255, 92, 110, 22), blur=72)
    draw = ImageDraw.Draw(img)
    scale = 0.58 if og else 1.0

    text_center(draw, (width / 2, 36 if og else 56), "SILENT ROTATION  ·  6 MODELS  ·  246 TRANSCRIPTS", font(HN, int(18 * scale) if og else 18, 0), CYAN)
    text_center(draw, (width / 2, 78 if og else 122), "Tests went green. Production was void.", font(HN, 32 if og else 54, 0), TEXT)
    text_center(draw, (width / 2, 118 if og else 186), "Three agents. One repo. The live signing key lived only in memory.", font(HN, 16 if og else 24, 0), MUTED)

    gap = 18 if og else 28
    margin = 36 if og else 64
    card_w = (width - margin * 2 - gap * 2) / 3
    card_h = 280 if og else 430
    card_y = 148 if og else 250
    fonts = {
        "name": font(HN, 13 if og else 20, 0),
        "small": font(HN, 11 if og else 15, 0),
        "num": font(MENLO, 44 if og else 72, 1),
        "frac": font(MENLO, 14 if og else 22, 0),
        "cap": font(HN, 13 if og else 18, 0),
    }
    cards = [
        ("NO MEMORY", 21, 25, 0, 4, WRONG, False),
        ("DENSE COSINE RAG", 12, 23, 4, 7, ORANGE, False),
        ("VESTIGE", 0, 23, 20, 3, MINT, True),
    ]
    for i, spec in enumerate(cards):
        draw_card(draw, margin + i * (card_w + gap), card_y, card_w, card_h, *spec, fonts)

    if og:
        text_center(draw, (width / 2, height - 22), "Pass = green tests AND production replay AND the correct key", font(HN, 12, 0), DIM)
        return img.convert("RGB")

    panel_y = card_y + card_h + 40
    panel_h = height - panel_y - 56
    rounded(draw, (margin, panel_y, width - margin, panel_y + panel_h), 28, CARD, outline=LINE, width=1)
    text_center(draw, (width / 2, panel_y + 36), "FIRST MEMORY CALL CONTAINED THE CORRECT KEY", font(HN, 20, 0), MUTED)
    rows = [
        ("VESTIGE  ·  NO QUERY", 65, 65, MINT),
        ("EVERY QUERY-BASED ARM", 7, 114, WRONG),
    ]
    inner_x = margin + 48
    inner_w = width - margin * 2 - 280
    bar_label = font(MENLO, 18, 1)
    for i, (label, hit, n, color) in enumerate(rows):
        y = panel_y + 84 + i * 78
        draw.text((inner_x, y), label, font=bar_label, fill=TEXT)
        track_y = y + 32
        rounded(draw, (inner_x, track_y, inner_x + inner_w, track_y + 16), 8, (255, 255, 255, 16))
        fill = max(10, int(inner_w * hit / n))
        rounded(draw, (inner_x, track_y, inner_x + fill, track_y + 16), 8, color)
        draw.text((inner_x + inner_w + 24, y + 8), pct_label(hit, n), font=bar_label, fill=color)

    text_center(
        draw,
        (width / 2, height - 28),
        "Pass = green tests  AND  production replay  AND  the correct key.   Ablation: bus-on-RAG 0/5  ·  no causal edge 0/5  ·  anchor-only 0/5",
        font(HN, 15, 0),
        DIM,
    )
    return img.convert("RGB")


def main():
    hero = render(2400, 1350, og=False)
    og = render(1200, 630, og=True)
    hero.save(OUT_HERO, "PNG", optimize=True)
    og.save(OUT_OG, "PNG", optimize=True)
    print(f"wrote {OUT_HERO} {hero.size}")
    print(f"wrote {OUT_OG} {og.size}")


if __name__ == "__main__":
    main()
