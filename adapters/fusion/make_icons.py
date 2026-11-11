"""Icons of the FEA Studio button for Fusion (16, 32 and @2x): a light cube with coloured stress bands and an arrow out."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "FEAStudio" / "resources" / "SendToFEA"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = [(38, 76, 200), (40, 150, 230), (60, 200, 120), (240, 210, 40), (240, 120, 30), (215, 40, 40)]


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def icon(size):
    S = 256
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # isometric cube
    c, r = (118, 140), 92
    top = (c[0], c[1] - r)
    tl, tr = (c[0] - 0.866 * r, c[1] - 0.5 * r), (c[0] + 0.866 * r, c[1] - 0.5 * r)
    bl, br = (c[0] - 0.866 * r, c[1] + 0.5 * r), (c[0] + 0.866 * r, c[1] + 0.5 * r)
    bot = (c[0], c[1] + r)
    # top face light grey, side faces with the stress bands (from left blue → right red)
    d.polygon([top, tr, c, tl], fill=(236, 238, 242), outline=(70, 80, 95))
    n = len(BANDS)
    for k, col in enumerate(BANDS):
        t0, t1 = k / n, (k + 1) / n
        if t1 <= 0.5:      # left face
            a0, a1 = t0 / 0.5, t1 / 0.5
            poly = [lerp(tl, c, a0), lerp(tl, c, a1), lerp(bl, bot, a1), lerp(bl, bot, a0)]
        else:
            a0, a1 = (t0 - 0.5) / 0.5, (t1 - 0.5) / 0.5
            poly = [lerp(c, tr, a0), lerp(c, tr, a1), lerp(bot, br, a1), lerp(bot, br, a0)]
        d.polygon(poly, fill=col)
    w = 7
    for a, b in [(tl, c), (c, tr), (c, bot), (tl, bl), (bl, bot), (bot, br), (br, tr), (tl, top), (top, tr)]:
        d.line([a, b], fill=(50, 58, 70), width=w)
    # mesh (diagonals on the side faces)
    for a, b in [(tl, bot), (c, br)]:
        d.line([a, b], fill=(255, 255, 255, 170), width=3)
    # the "send" arrow top right
    ax, ay = 190, 52
    d.line([(150, 92), (ax, ay)], fill=(25, 110, 210), width=18)
    d.polygon([(ax + 22, ay - 22), (ax - 26, ay - 14), (ax + 14, ay + 26)], fill=(25, 110, 210))
    return im.resize((size, size), Image.LANCZOS)


for name, s in [("16x16.png", 16), ("16x16@2x.png", 32), ("32x32.png", 32), ("32x32@2x.png", 64)]:
    icon(s).save(OUT / name)
icon(128).save(OUT.parent.parent.parent / "preview_128.png") if "--preview" in sys.argv else None
print("ok")
