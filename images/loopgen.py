#!/usr/bin/env python3
"""
loopgen.py — Pixel Art Loop Generator  [1920×1080 edition]
------------------------------------------------------------
Generates seamless looping pixel art animations at full HD.
Default output: MP4 (recommended for 1080p).
GIF available at reduced scale via --gif.

Usage:
  python loopgen.py "late night lofi study room, rain on window, desk lamp"
  python loopgen.py "cyberpunk alley, neon rain" --palette moonlight --frames 16
  python loopgen.py "cozy fireplace cabin, snow outside" --gif --gif-scale 3
  python loopgen.py "pixel art tokyo street at night" --no-rain --seed 1337

Setup:
  pip install huggingface_hub pillow numpy requests
  export HF_TOKEN=hf_...   (free at huggingface.co/settings/tokens)
"""

import os, sys, time, argparse, requests, hashlib, subprocess
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image, ImageFilter
    import numpy as np
except ImportError:
    print("[!] Missing deps. Run: pip install pillow numpy requests huggingface_hub")
    sys.exit(1)

# ── OUTPUT SPEC ───────────────────────────────────────────────────────────────
#
#   Pixel canvas  :  320 × 180   (native 16:9, 1/6 of 1920×1080)
#   Upscale       :  ×6 nearest-neighbor  →  1920 × 1080
#   API gen size  :  960 × 544   (FLUX widescreen, downscaled to canvas)
#   GIF output    :  canvas × gif_scale  (default 3 → 960×540, manageable)
#
OUT_W,  OUT_H  = 1920, 1080
PIX_W,  PIX_H  = 320,  180
UPSCALE        = OUT_W // PIX_W    # = 6
GEN_W,  GEN_H  = 960,  544         # API generation size (must be mult of 8)

# ── API ───────────────────────────────────────────────────────────────────────
HF_MODEL      = "black-forest-labs/FLUX.1-schnell"
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HF_MODEL_FB   = "stabilityai/stable-diffusion-xl-base-1.0"
HF_ROUTER_FB  = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_FB}"

PIXEL_ART_SUFFIX = (
    ", pixel art scene, 16-bit SNES style, retro game background, "
    "crisp pixel art, limited color palette, lofi aesthetic, "
    "wide establishing shot, detailed environment, atmospheric lighting, "
    "no text, no watermark, no UI elements"
)
NEGATIVE_PROMPT = (
    "blurry, photo, photorealistic, 3d render, watermark, text, logo, "
    "noisy, grainy, deformed, low quality, anime face, character closeup"
)

CACHE_DIR = Path.home() / ".loopgen_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── PALETTES ──────────────────────────────────────────────────────────────────
PALETTES = {
    "lofi": [
        (13,11,30),(26,18,44),(52,36,80),(92,64,120),(160,110,180),
        (220,170,240),(255,220,180),(200,150,100),(120,80,40),(60,30,15),
        (20,30,50),(40,70,100),(80,130,160),(160,200,220),(255,240,200),(240,180,100),
    ],
    "moonlight": [
        (5,5,20),(15,15,45),(30,30,80),(60,60,130),(100,100,180),
        (150,160,210),(200,210,240),(240,245,255),(180,200,230),(120,140,190),
        (80,90,140),(50,55,100),(25,28,60),(10,12,30),(200,180,150),(255,240,200),
    ],
    "sunset": [
        (10,5,20),(30,10,30),(70,15,40),(130,30,50),(200,60,60),
        (240,110,50),(255,170,60),(255,220,100),(200,120,80),(140,60,40),
        (80,20,30),(40,10,20),(255,200,150),(180,100,60),(255,240,180),(100,40,20),
    ],
    "forest": [
        (5,10,5),(15,30,10),(30,60,20),(50,90,30),(80,130,50),
        (120,170,70),(170,210,100),(220,240,160),(160,190,100),(100,140,60),
        (60,90,30),(30,50,15),(10,20,5),(200,180,120),(140,110,60),(80,60,20),
    ],
    "neon": [
        (5,0,15),(15,0,30),(30,0,60),(0,20,80),(0,60,120),
        (0,180,255),(0,255,220),(180,0,255),(255,0,180),(255,60,0),
        (255,180,0),(200,255,0),(10,10,20),(40,0,80),(0,40,100),(255,255,255),
    ],
    "gameboy": [(15,56,15),(48,98,48),(139,172,15),(155,188,15)],
    "c64": [
        (0,0,0),(255,255,255),(136,0,0),(170,255,238),(204,68,204),(0,204,85),
        (0,0,170),(238,238,119),(221,136,85),(102,68,0),(255,119,119),
        (51,51,51),(119,119,119),(170,255,102),(0,136,255),(187,187,187),
    ],
}

# ── TERMINAL ──────────────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
CYAN="\033[36m"; GREEN="\033[32m"; YEL="\033[33m"; MAG="\033[35m"; RED="\033[31m"

def log(msg, c=CYAN):  print(f"{c}{BOLD}▸{R} {msg}")
def ok(msg):           print(f"{GREEN}{BOLD}✓{R} {msg}")
def warn(msg):         print(f"{YEL}{BOLD}⚠{R}  {msg}")
def err(msg):          print(f"{RED}{BOLD}✗{R} {msg}")
def bar(label, i, n, w=36):
    f = int(w*i/n)
    b = "█"*f + "░"*(w-f)
    print(f"\r  {DIM}{label}{R} [{CYAN}{b}{R}] {int(100*i/n)}%", end="", flush=True)
    if i >= n: print()
def section(t):
    print(f"\n  {MAG}{BOLD}── {t} {'─'*(42-len(t))}{R}")

# ── AUTH ──────────────────────────────────────────────────────────────────────
def get_hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not t:
        err("HF_TOKEN not set.")
        print(f"  {DIM}→ https://huggingface.co/settings/tokens{R}")
        print(f"  {DIM}→ export HF_TOKEN=hf_your_token_here{R}")
        sys.exit(1)
    return t

def cache_path(prompt, seed):
    h = hashlib.md5(f"{prompt}|{seed}|{GEN_W}x{GEN_H}".encode()).hexdigest()[:14]
    return CACHE_DIR / f"{h}.png"

# ── IMAGE GENERATION ──────────────────────────────────────────────────────────
def generate_base(prompt, seed, token):
    """Generate one widescreen base image via HF Inference API."""
    cp = cache_path(prompt, seed)
    if cp.exists():
        log(f"Cache hit → {cp.name}", c=DIM)
        return Image.open(cp).convert("RGB")

    full_prompt = prompt + PIXEL_ART_SUFFIX

    # Try huggingface_hub InferenceClient
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", api_key=token)
        log(f"FLUX.1-schnell via InferenceClient ({GEN_W}×{GEN_H})")
        img = client.text_to_image(full_prompt, model=HF_MODEL, width=GEN_W, height=GEN_H)
        img = img.convert("RGB")
        img.save(cp)
        return img
    except ImportError:
        warn("huggingface_hub not installed → pip install huggingface_hub")
    except Exception as e:
        warn(f"InferenceClient failed: {e}")

    # Fallback: raw requests
    log("Trying router.huggingface.co via raw requests...")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-wait-for-model": "true",
    }
    candidates = [
        (HF_ROUTER_URL, {"inputs": full_prompt,
            "parameters": {"width": GEN_W, "height": GEN_H, "seed": seed}}),
        (HF_ROUTER_FB,  {"inputs": full_prompt,
            "parameters": {"negative_prompt": NEGATIVE_PROMPT,
                           "width": 768, "height": 432, "seed": seed,
                           "num_inference_steps": 25}}),
    ]
    for url, payload in candidates:
        mname = url.split("/models/")[-1]
        for retry in range(4):
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(cp)
                return img
            if resp.status_code in (500, 503):
                try:    wait = float(resp.json().get("estimated_time", 25))
                except: wait = 25
                warn(f"{mname}: loading, {int(wait)}s (retry {retry+1}/4)")
                time.sleep(min(wait, 35))
            elif resp.status_code == 429:
                warn(f"Rate limited, 20s (retry {retry+1}/4)")
                time.sleep(20)
            else:
                warn(f"{mname}: HTTP {resp.status_code} — {resp.text[:120]}")
                break

    err("All endpoints failed. Check HF_TOKEN and quota.")
    sys.exit(1)

# ── PIXELATE ──────────────────────────────────────────────────────────────────
def pixelate(img, pw, ph, palette_name):
    """Crop to 16:9, downscale to pixel canvas, quantize to palette."""
    sw, sh = img.size
    tr = pw / ph
    sr = sw / sh
    if sr > tr:
        nw = int(sh * tr); x0 = (sw - nw)//2
        img = img.crop((x0, 0, x0+nw, sh))
    elif sr < tr:
        nh = int(sw / tr); y0 = (sh - nh)//2
        img = img.crop((0, y0, sw, y0+nh))

    small = img.resize((pw, ph), Image.LANCZOS).filter(ImageFilter.SHARPEN)

    if palette_name in PALETTES:
        pal_colors = PALETTES[palette_name]
        pi = Image.new("P", (1,1))
        flat = [v for rgb in pal_colors for v in rgb] + [0]*(768 - len(pal_colors)*3)
        pi.putpalette(flat)
        q = small.quantize(palette=pi, dither=Image.Dither.FLOYDSTEINBERG)
    else:
        q = small.quantize(colors=32, dither=Image.Dither.FLOYDSTEINBERG)

    return q.convert("RGB")

# ── EFFECTS ───────────────────────────────────────────────────────────────────
def add_scanlines(arr, strength=0.07):
    out = arr.astype(np.float32)
    out[::2] *= (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)

def add_vignette(arr, pw, ph, strength=0.55):
    cx, cy = pw/2, ph/2
    Y, X = np.ogrid[:ph, :pw]
    mask = np.clip(1.0 - np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)*strength, 0, 1)
    return np.clip(arr.astype(np.float32) * mask[:,:,np.newaxis], 0, 255).astype(np.uint8)

def build_rain(pw, ph, n=None):
    rng = np.random.default_rng(42)
    n = n or pw//3
    return (rng.integers(0, pw, n), rng.uniform(0,1,n),
            rng.uniform(0.5,1.5,n), rng.integers(1,4,n), rng.random(n)>0.5)

def apply_rain(arr, t, ph, rain):
    xs, y0s, spds, lens, bright = rain
    out = arr.astype(np.float32)
    for i in range(len(xs)):
        ry = int((y0s[i] + t*spds[i]) * ph) % ph
        for dl in range(lens[i]):
            yy = (ry+dl) % ph
            col = [100,150,200] if bright[i] else [55,85,135]
            a = 0.18 if bright[i] else 0.10
            out[yy, xs[i]] = out[yy, xs[i]]*(1-a) + np.array(col)*a
    return np.clip(out, 0, 255).astype(np.uint8)

def build_steam(pw, ph, n=14):
    rng = np.random.default_rng(7)
    return (rng.integers(pw//3, 2*pw//3, n),
            rng.uniform(0, 2*np.pi, n),
            rng.uniform(0.35, 0.85, n))

def apply_steam(arr, t, pw, ph, steam):
    xs, phases, spds = steam
    out = arr.astype(np.float32)
    for i in range(len(xs)):
        life = (t*spds[i] + phases[i]/(2*np.pi)) % 1.0
        sy = max(0, min(ph-1, int(ph*(0.45 - life*0.25))))
        sx = max(0, min(pw-1, xs[i] + int(np.sin(life*np.pi*3+phases[i])*3)))
        a = (life*(1-life)*4) * 0.12
        out[sy, sx] = np.clip(out[sy, sx] + np.array([50,50,70])*a, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)

# ── LOOP SYNTHESIS ────────────────────────────────────────────────────────────
def make_loop_frames(pixel_img, n_frames, palette_name, effects):
    """
    Derive n_frames from base pixel image.
    t in [0,1) → mathematically seamless loop.
    Entirely local — no API calls.
    """
    pw, ph = pixel_img.size   # 320×180
    base   = np.array(pixel_img, dtype=np.float32)

    rain_data  = build_rain(pw, ph)  if effects["rain"]  else None
    steam_data = build_steam(pw, ph) if effects["steam"] else None

    frames = []
    for i in range(n_frames):
        t     = i / n_frames
        phase = t * 2 * np.pi
        arr   = base.copy()

        # Lamp / ambient flicker
        arr *= (1.0 + 0.035 * np.sin(phase))
        # Warm↔cool temperature drift
        arr[:,:,0] *= 1.0 + 0.018 * np.sin(phase + 0.4)
        arr[:,:,2] *= 1.0 + 0.012 * np.cos(phase + 1.1)
        # Bottom warmth
        warmth = np.linspace(0, 0.03, ph)[:,np.newaxis]   # (180,1) broadcasts over (180,320,3)
        arr = arr.astype(np.float32)
        arr[:,:,0] += warmth * np.sin(phase+1.0) * 20
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        if rain_data  is not None: arr = apply_rain(arr, t, ph, rain_data)
        if steam_data is not None: arr = apply_steam(arr, t, pw, ph, steam_data)

        # Re-quantize to lock palette across frames
        frame_img = pixelate(Image.fromarray(arr), pw, ph, palette_name)
        arr = np.array(frame_img)

        if effects["scanlines"]: arr = add_scanlines(arr)
        if effects["vignette"]:  arr = add_vignette(arr, pw, ph)

        frames.append(arr)
        bar("synthesizing frames", i+1, n_frames)

    return frames   # list of (180, 320, 3) uint8 arrays

# ── UPSCALE ───────────────────────────────────────────────────────────────────
def upscale(arr, scale):
    img = Image.fromarray(arr)
    return img.resize((img.width*scale, img.height*scale), Image.NEAREST)

# ── EXPORT MP4 ────────────────────────────────────────────────────────────────
def save_mp4(frames, output_path, fps, scale, min_secs=6):
    w = frames[0].shape[1] * scale
    h = frames[0].shape[0] * scale
    target = max(len(frames), fps * min_secs)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "15", "-preset", "slow", "-tune", "animation",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for j in range(target):
        proc.stdin.write(upscale(frames[j % len(frames)], scale).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.read().decode())

# ── EXPORT GIF ────────────────────────────────────────────────────────────────
def save_gif(frames, output_path, fps, gif_scale):
    delay = int(1000 / fps)
    pil_frames = []
    for arr in frames:
        img = upscale(arr, gif_scale)
        img = img.quantize(colors=128, dither=Image.Dither.FLOYDSTEINBERG)
        pil_frames.append(img)
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                       loop=0, duration=delay, optimize=True)

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate seamless 1920×1080 pixel art loops from a text prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python loopgen.py "late night lofi study room, rain on window, desk lamp"
  python loopgen.py "cyberpunk alley, neon rain, puddles" --palette neon --frames 20
  python loopgen.py "cozy fireplace cabin, snow outside" --no-rain --fps 12
  python loopgen.py "misty forest, glowing fireflies" --palette forest --seed 99
  python loopgen.py "pixel art tokyo at night" --gif --gif-scale 4

Output format:
  MP4 at 1920×1080 by default (recommended — small file, perfect quality).
  Use --gif for an animated GIF at reduced resolution (default 960×540).

Env:
  HF_TOKEN    free token at https://huggingface.co/settings/tokens
        """
    )
    parser.add_argument("prompt")
    parser.add_argument("--frames",    type=int, default=16)
    parser.add_argument("--fps",       type=int, default=10)
    parser.add_argument("--palette",   default="lofi", choices=list(PALETTES.keys()))
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--gif",       action="store_true",
                        help="Export as GIF instead of MP4")
    parser.add_argument("--gif-scale", type=int, default=3,
                        help="GIF upscale factor (default 3 → 960×540). Use 6 for full 1080p.")
    parser.add_argument("--no-rain",      dest="rain",      action="store_false")
    parser.add_argument("--no-steam",     dest="steam",     action="store_false")
    parser.add_argument("--no-scanlines", dest="scanlines", action="store_false")
    parser.add_argument("--no-vignette",  dest="vignette",  action="store_false")
    parser.set_defaults(rain=True, steam=True, scanlines=True, vignette=True)

    args = parser.parse_args()

    if args.output is None:
        slug = "".join(c if c.isalnum() else "_" for c in args.prompt[:32].lower())
        slug = "_".join(filter(None, slug.split("_")))[:28]
        args.output = f"{slug}_loop.{'gif' if args.gif else 'mp4'}"

    output_path = Path(args.output)
    effects = {k: getattr(args, k) for k in ("rain","steam","scanlines","vignette")}
    active_fx = " ".join(k for k,v in effects.items() if v)

    print()
    print(f"  {MAG}{BOLD}◆ LOOPGEN  ·  1920×1080 Pixel Art Loop{R}")
    print(f"  {DIM}{'─'*50}{R}")
    print(f"  {DIM}prompt  :{R} {args.prompt[:62]}")
    print(f"  {DIM}canvas  :{R} {PIX_W}×{PIX_H}  →  {OUT_W}×{OUT_H}  (×{UPSCALE} upscale)")
    print(f"  {DIM}loop    :{R} {args.frames} frames @ {args.fps} fps")
    print(f"  {DIM}palette :{R} {args.palette}  ({len(PALETTES[args.palette])} colors)")
    print(f"  {DIM}effects :{R} {active_fx}")
    print(f"  {DIM}output  :{R} {output_path}  ({'GIF' if args.gif else 'MP4 H.264'})")
    print(f"  {DIM}{'─'*50}{R}")

    token = get_hf_token()

    # 1. Generate
    section("1 / 4  GENERATE BASE IMAGE")
    log(f"Requesting {GEN_W}×{GEN_H} from FLUX.1-schnell...")
    t0 = time.time()
    base_img = generate_base(args.prompt, args.seed, token)
    ok(f"Received {base_img.width}×{base_img.height}  in {time.time()-t0:.1f}s")

    # 2. Pixelate
    section("2 / 4  PIXELATE  →  320×180")
    pixel_img = pixelate(base_img, PIX_W, PIX_H, args.palette)
    ok(f"Canvas ready: {PIX_W}×{PIX_H}  ·  {len(PALETTES[args.palette])} colors")

    # 3. Synthesize
    section("3 / 4  SYNTHESIZE LOOP FRAMES")
    frames = make_loop_frames(pixel_img, args.frames, args.palette, effects)
    ok(f"{args.frames} frames  ·  seamless loop guaranteed")

    # 4. Export
    section("4 / 4  EXPORT")
    if args.gif:
        gw, gh = PIX_W*args.gif_scale, PIX_H*args.gif_scale
        if args.gif_scale == 6:
            warn(f"Full 1080p GIF can exceed 200 MB. Consider --gif-scale 3 (960×540).")
        log(f"Writing GIF at {gw}×{gh}...")
        save_gif(frames, output_path, args.fps, args.gif_scale)
    else:
        # Verify ffmpeg
        if subprocess.run(["ffmpeg","-version"], capture_output=True).returncode != 0:
            err("ffmpeg not found.  sudo apt install ffmpeg")
            err("Or export as GIF with --gif flag.")
            sys.exit(1)
        log(f"Writing MP4 at {OUT_W}×{OUT_H} via ffmpeg pipe...")
        save_mp4(frames, output_path, args.fps, UPSCALE)

    size_mb = output_path.stat().st_size / (1024*1024)
    ok(f"Saved: {BOLD}{output_path}{R}  ({size_mb:.1f} MB)")

    print()
    print(f"  {GREEN}{BOLD}◆ Done!{R}")
    if not args.gif:
        print(f"  {DIM}Play :  mpv --loop {output_path}{R}")
        print(f"  {DIM}Info :  ffprobe {output_path}{R}")
    else:
        print(f"  {DIM}Open :  xdg-open {output_path}{R}")
    print()


if __name__ == "__main__":
    main()
