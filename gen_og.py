#!/usr/bin/env python3
"""Render og.png (1200x630) for The Fine-Tuning Index — industrial forge card. Pillow only."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _font(paths, size):
    from PIL import ImageFont
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main() -> int:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("Pillow not available — skipping og.png")
        return 0
    try:
        data = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
        count, cats = data.get("count", 0), len(data.get("categories", []))
    except Exception:
        count, cats = 0, 0

    W, H = 1200, 630
    bg, ink, ember, hot, muted = (21, 18, 13), (243, 236, 223), (255, 106, 0), (255, 196, 0), (165, 154, 133)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    # hazard stripes top + bottom
    for i in range(-2, 40):
        d.line([(i * 40, 0), (i * 40 - 30, 30)], fill=(40, 32, 18), width=12)
        d.line([(i * 40, H), (i * 40 - 30, H - 30)], fill=(40, 32, 18), width=12)

    bold = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
    mono = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]
    f_kick = _font(mono, 24)
    f_h1 = _font(bold, 96)
    f_stat = _font(mono, 28)

    d.rectangle([70, 96, 94, 120], fill=ember)
    d.text((108, 92), "THE FINE-TUNING INDEX", font=f_kick, fill=ember)
    d.text((68, 180), "FORGE YOUR", font=f_h1, fill=ink)
    d.text((68, 290), "OWN ", font=f_h1, fill=ink)
    w = d.textlength("OWN ", font=f_h1)
    d.text((68 + w, 290), "MODEL.", font=f_h1, fill=ember)
    d.text((70, 470), f"{count} tools  ·  {cats} categories  ·  ranked daily by GitHub momentum",
           font=f_stat, fill=muted)
    img.save(os.path.join(HERE, "og.png"))
    print(f"wrote og.png ({count} tools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
