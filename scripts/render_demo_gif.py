"""Render docs/demo.gif offline from recorded episodes (no browser needed).

Three panels play the SAME start (one seed) flown by three checkpoints of one training run, side by side:
an early checkpoint, a middle one, and the final policy. Frames come from the static-demo JSON written by
scripts/export_static_demo.py, so every frame is real simulator output, not an animation.

    uv run python scripts/render_demo_gif.py                       # defaults: frontend/demo, seed 20000
    uv run python scripts/render_demo_gif.py --run exp_fixed_s0 --seed 20003 --out docs/demo.gif

Needs Pillow (installed with the `plots` extra, via matplotlib).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (8, 12, 22)
GRID = (20, 28, 44)
GROUND = (22, 32, 52)
PAD = (60, 220, 160)
TRAIL = (245, 170, 60)
TEXT = (220, 228, 240)
DIM = (120, 132, 156)
LAND = (70, 230, 150)
CRASH = (255, 90, 90)
FLAME = (255, 190, 70)

SS = 2  # supersampling factor for anti-aliased edges


def font(size: int) -> ImageFont.ImageFont:
    for name in ("Menlo.ttc", "Helvetica.ttc", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size * SS)
        except OSError:
            continue
    return ImageFont.load_default(size=size * SS)


def load_episode(demo: Path, run: str, ckpt: str, seed: int) -> dict:
    return json.loads((demo / f"episode_{run}__{ckpt}__{seed}.json").read_text())


class View:
    """World -> panel pixel mapping for one panel (pad centred, fixed window)."""

    def __init__(self, w: int, h: int, x_half: float, y_top: float, ground_px: int):
        self.w, self.h, self.x_half, self.y_top, self.ground_px = w * SS, h * SS, x_half, y_top, ground_px * SS
        self.scale = min(self.w / (2 * x_half), (self.h - self.ground_px) / y_top)

    def p(self, x: float, y: float) -> tuple[float, float]:
        return self.w / 2 + x * self.scale, self.h - self.ground_px - y * self.scale


def rocket_polygon(view: View, x: float, y: float, angle: float, size: float = 8.0):
    """Body as a pointed capsule; angle is counter-clockwise positive (docs/environment.md)."""
    pts = [(0, size), (size * 0.28, size * 0.35), (size * 0.28, -size * 0.45), (-size * 0.28, -size * 0.45),
           (-size * 0.28, size * 0.35)]
    ca, sa = math.cos(angle), math.sin(angle)
    return [view.p(x + px * ca - py * sa, y + px * sa + py * ca) for px, py in pts]


def draw_panel(ep: dict, k: int, hold: int, label: str, w: int, h: int, fonts) -> Image.Image:
    fr, meta = ep["frames"], ep["meta"]
    n = len(fr)
    i = min(k, n - 1)
    f = fr[i]
    view = View(w, h, x_half=45.0, y_top=128.0, ground_px=46)
    img = Image.new("RGB", (view.w, view.h), BG)
    d = ImageDraw.Draw(img)

    for gx in range(-40, 41, 20):  # faint grid
        a, b = view.p(gx, 0), view.p(gx, 128)
        d.line([a, b], fill=GRID, width=SS)
    for gy in range(0, 121, 20):
        a, b = view.p(-45, gy), view.p(45, gy)
        d.line([a, b], fill=GRID, width=SS)
    gy0 = view.p(0, 0)[1]
    d.rectangle([0, gy0, view.w, view.h], fill=GROUND)
    pa, pb = view.p(-meta["world"]["pad_half_width"], 0), view.p(meta["world"]["pad_half_width"], 0)
    d.rectangle([pa[0], gy0 - 2 * SS, pb[0], gy0 + 4 * SS], fill=PAD)

    start = max(0, i - 90)  # fading trail
    for j in range(start + 1, i + 1):
        t = (j - start) / max(1, i - start)
        col = tuple(int(BG[c] + (TRAIL[c] - BG[c]) * t * 0.9) for c in range(3))
        d.line([view.p(fr[j - 1]["x"], fr[j - 1]["y"]), view.p(f["x"] if j == i else fr[j]["x"],
                                                                f["y"] if j == i else fr[j]["y"])],
               fill=col, width=SS * 2)

    done = k >= n - 1
    outcome = ep["outcome"]
    if done and outcome == "crashed":
        cx, cy = view.p(f["x"], 1.5)
        age = min(1.0, (k - (n - 1)) / max(1, hold * 0.35))
        r = (4 + 16 * age) * SS
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=CRASH, width=2 * SS)
        d.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], fill=(200, 70, 60))
    else:
        poly = rocket_polygon(view, f["x"], f["y"], f["angle"])
        act = f["action"]
        ca, sa = math.cos(f["angle"]), math.sin(f["angle"])
        if act == 1 and f["fuel"] > 0:  # main engine flame, out of the bottom
            fl = [(-1.7, -3.6), (1.7, -3.6), (0, -12.0)]
            d.polygon([view.p(f["x"] + px * ca - py * sa, f["y"] + px * sa + py * ca) for px, py in fl],
                      fill=FLAME)
        if act in (2, 3) and f["fuel"] > 0:  # side thrusters: small puff near the nose
            sgn = 1 if act == 2 else -1  # LEFT fires the counter-clockwise couple
            px0, py0 = 2.4 * sgn, 4.8
            q = [view.p(f["x"] + (px0 + dx) * ca - (py0 + dy) * sa, f["y"] + (px0 + dx) * sa + (py0 + dy) * ca)
                 for dx, dy in ((0, 0.8), (0, -0.8), (4.2 * sgn, 0))]
            d.polygon(q, fill=FLAME)
        d.polygon(poly, fill=(226, 232, 242), outline=(150, 160, 180))
        cx, cy = view.p(f["x"], f["y"] + 1.6)
        d.ellipse([cx - 3 * SS, cy - 3 * SS, cx + 3 * SS, cy + 3 * SS], fill=(60, 110, 170))
        if done and outcome == "landed":
            lx, ly = view.p(f["x"], 0)
            age = min(1.0, (k - (n - 1)) / max(1, hold * 0.3))
            r = (6 + 22 * age) * SS
            d.ellipse([lx - r, ly - r * 0.35, lx + r, ly + r * 0.35], outline=LAND, width=2 * SS)

    # header + footer text
    d.text((10 * SS, 8 * SS), label, font=fonts["h"], fill=TEXT)
    d.text((10 * SS, 24 * SS), f"update {meta['update']}" if meta["update"] else "", font=fonts["s"], fill=DIM)
    fuel_w = 90 * SS
    fx, fy = view.w - 150 * SS, 10 * SS
    d.text((fx, fy), "fuel", font=fonts["s"], fill=DIM)
    d.rectangle([fx + 32 * SS, fy + 2 * SS, fx + 32 * SS + fuel_w, fy + 12 * SS], outline=DIM, width=SS)
    if f["fuel"] > 0:
        bar = (255, 190, 70) if f["fuel"] > 0.25 else CRASH
        d.rectangle([fx + 32 * SS, fy + 2 * SS, fx + 32 * SS + fuel_w * f["fuel"], fy + 12 * SS], fill=bar)
    else:
        d.text((fx + 38 * SS, fy), "EMPTY", font=fonts["s"], fill=CRASH)
    status, col = ("IN FLIGHT", DIM)
    if done:
        status, col = (("LANDED", LAND) if outcome == "landed" else ("CRASHED", CRASH))
    d.text((10 * SS, view.h - 24 * SS), status, font=fonts["h"], fill=col)
    d.text((view.w - 130 * SS, view.h - 22 * SS), f"vy {f['vy']:+.1f} m/s", font=fonts["s"], fill=DIM)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir", default="frontend/demo")
    ap.add_argument("--run", default="exp_fixed_s0")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--checkpoints", nargs=3, default=["ckpt_00100", "ckpt_00500", "final"])
    ap.add_argument("--labels", nargs=3, default=["Early", "Middle", "Final"])
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--panel-w", type=int, default=400)
    ap.add_argument("--panel-h", type=int, default=440)
    ap.add_argument("--step", type=int, default=7, help="simulator steps per GIF frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold", type=int, default=45, help="frames to hold on the last frame")
    ap.add_argument("--colors", type=int, default=256)
    args = ap.parse_args()

    demo = Path(args.demo_dir)
    eps = [load_episode(demo, args.run, c, args.seed) for c in args.checkpoints]
    fonts = {"h": font(15), "s": font(11), "t": font(13)}
    longest = max(len(e["frames"]) for e in eps)
    n_frames = math.ceil(longest / args.step) + args.hold
    w, h = args.panel_w, args.panel_h
    gap, top = 8, 34

    frames = []
    for fi in range(n_frames):
        k = fi * args.step
        canvas = Image.new("RGB", ((3 * w + 2 * gap) * SS, (h + top) * SS), BG)
        cd = ImageDraw.Draw(canvas)
        title = f"Same start (seed {args.seed}), three checkpoints of one PPO training run"
        cd.text((10 * SS, 9 * SS), title, font=fonts["t"], fill=TEXT)
        for pi, (ep, lab) in enumerate(zip(eps, args.labels)):
            panel = draw_panel(ep, k, args.hold * args.step, lab, w, h, fonts)
            canvas.paste(panel, (pi * (w + gap) * SS, top * SS))
        small = canvas.resize((canvas.width // SS, canvas.height // SS), Image.LANCZOS)
        frames.append(small)

    # One shared palette built from a montage of early, middle and last frames, so red/green end states survive.
    sample = [frames[0], frames[len(frames) // 3], frames[2 * len(frames) // 3], frames[-1]]
    montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for si, sf in enumerate(sample):
        montage.paste(sf, (0, si * sf.height))
    pal = montage.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / args.fps), loop=0, optimize=True,
              disposal=1)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(q)} frames, {q[0].size[0]}x{q[0].size[1]})")


if __name__ == "__main__":
    main()
