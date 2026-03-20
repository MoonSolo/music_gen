"""
generate_thumbnail.py — Generate YouTube thumbnails via ComfyUI API
Requires ComfyUI running: cd ~/ComfyUI && source venv/bin/activate && python main.py --listen 127.0.0.1 --port 7860

Usage:
  python3 scripts/generate_thumbnail.py --video_name lofi_commute --preset rainy_night
  python3 scripts/generate_thumbnail.py --video_name lofi_commute --prompt "anime girl at desk, rainy window"
  python3 scripts/generate_thumbnail.py --video_name lofi_commute --preset study_desk --variants 4
  python3 scripts/generate_thumbnail.py --video_name lofi_vol1 --all_presets

Presets: rainy_night, study_desk, rooftop_sunset, midnight_cafe,
         gaming_room, dark_3am, morning_light, open_world

Output: thumbnails/<video_name>/thumbnail_<preset>_v1.png ...
"""

import argparse
import json
import sys
import time
import uuid
import random
import requests
import urllib.parse
from pathlib import Path


COMFY_URL    = "http://127.0.0.1:7860"
TIMEOUT_SEC  = 300
CLIENT_ID    = str(uuid.uuid4())
THUMB_WIDTH  = 1280
THUMB_HEIGHT = 720

STYLE_SUFFIX = (
    "anime style, Studio Ghibli inspired, lofi aesthetic, "
    "soft lighting, warm color palette, highly detailed, "
    "cinematic composition, cozy atmosphere, 4k, masterpiece, "
    "best quality, illustration"
)

NEGATIVE_PROMPT = (
    "nsfw, realistic, photograph, 3d render, ugly, deformed, "
    "blurry, low quality, bad anatomy, watermark, signature, "
    "text, logo, harsh lighting, oversaturated, cartoon, chibi"
)

PRESETS = {
    "rainy_night": (
        "anime girl sitting by window at night, rain drops on glass, "
        "neon signs reflecting in puddles below, warm indoor lamp light, "
        "city lights blurred in background, cozy bedroom, headphones around neck"
    ),
    "study_desk": (
        "anime student at wooden desk, open books and notebook, "
        "warm desk lamp glowing, pencil in hand, afternoon sunlight through window, "
        "plants on windowsill, cozy room, focused expression"
    ),
    "rooftop_sunset": (
        "anime character sitting on rooftop ledge, golden sunset sky, "
        "city skyline below, warm orange and pink light, "
        "legs dangling over edge, peaceful expression, soft wind"
    ),
    "midnight_cafe": (
        "cozy jazz cafe at night, warm amber lighting, "
        "small table with coffee cup, rain outside window, "
        "empty chairs, vintage decor, soft glow, intimate atmosphere"
    ),
    "gaming_room": (
        "anime teenager in bedroom at night, CRT monitor glowing blue, "
        "game controller on desk, posters on wall, "
        "warm lamp in corner, cozy cluttered desk, late night gaming session"
    ),
    "dark_3am": (
        "anime character lying in dark bedroom, 3am on digital clock, "
        "moonlight through curtains, ceiling stare, "
        "soft blue and purple tones, introspective mood, minimal lighting"
    ),
    "morning_light": (
        "anime character by sunny window, morning coffee steam, "
        "golden morning light streaming in, houseplants, "
        "soft white curtains blowing, peaceful sunday morning, warm tones"
    ),
    "open_world": (
        "anime character standing on hilltop, vast open landscape below, "
        "golden hour light, adventure mood, distant mountains, "
        "warm breeze, explorer bag, sense of freedom and scale"
    ),
}


def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YouTube thumbnails via ComfyUI")
    parser.add_argument("--video_name",  required=True)
    parser.add_argument("--preset",      default=None, choices=list(PRESETS.keys()))
    parser.add_argument("--prompt",      default=None)
    parser.add_argument("--variants",    default=2, type=int)
    parser.add_argument("--all_presets", action="store_true")
    parser.add_argument("--steps",       default=28, type=int)
    parser.add_argument("--cfg",         default=7.0, type=float)
    parser.add_argument("--seed",        default=-1, type=int)
    parser.add_argument("--checkpoint",  default="Counterfeit-V3.safetensors")
    parser.add_argument("--api_url",     default=COMFY_URL)
    return parser.parse_args()


def check_api(api_url):
    try:
        r = requests.get(f"{api_url}/system_stats", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def build_workflow(prompt, negative, checkpoint, steps, cfg, seed, width, height):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    return {
        "prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["1", 1]}
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model":        ["1", 0],
                    "positive":     ["2", 0],
                    "negative":     ["3", 0],
                    "latent_image": ["4", 0],
                    "seed":         seed,
                    "steps":        steps,
                    "cfg":          cfg,
                    "sampler_name": "dpm_2_ancestral",
                    "scheduler":    "karras",
                    "denoise":      1.0
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"images": ["6", 0], "filename_prefix": "thumbnail"}
            }
        },
        "client_id": CLIENT_ID
    }


def queue_and_wait(workflow, api_url):
    r = requests.post(f"{api_url}/prompt", json=workflow, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to queue prompt: {r.status_code} {r.text}")

    prompt_id = r.json()["prompt_id"]
    start     = time.time()

    while True:
        if time.time() - start > TIMEOUT_SEC:
            raise RuntimeError("Timed out waiting for generation")

        r = requests.get(f"{api_url}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                outputs = history[prompt_id]["outputs"]
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        return node_output["images"][0]
        time.sleep(1)


def download_image(img_info, api_url):
    params   = urllib.parse.urlencode({
        "filename":  img_info["filename"],
        "subfolder": img_info.get("subfolder", ""),
        "type":      img_info.get("type", "output")
    })
    response = requests.get(f"{api_url}/view?{params}", timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download image: {response.status_code}")
    return response.content


def main():
    args        = parse_args()
    total_start = time.time()

    thumb_dir = Path("thumbnails") / args.video_name
    thumb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Thumbnail Generator (ComfyUI)")
    print(f"  Video      : {args.video_name}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Steps      : {args.steps}  |  CFG: {args.cfg}")
    print(f"  Resolution : {THUMB_WIDTH}x{THUMB_HEIGHT}")
    print(f"  Output     : {thumb_dir}")
    print(f"{'='*55}\n")

    print(f"Checking ComfyUI API at {args.api_url}...")
    if not check_api(args.api_url):
        print(f"✗ ComfyUI not reachable at {args.api_url}")
        print(f"  Start it with:")
        print(f"  cd ~/ComfyUI && source venv/bin/activate && python main.py --listen 127.0.0.1 --port 7860")
        sys.exit(1)
    print(f"✓ ComfyUI is running\n")

    jobs = []
    if args.all_presets:
        for name, prompt in PRESETS.items():
            jobs.append({"name": name, "prompt": prompt, "variants": 1})
    elif args.prompt:
        jobs.append({"name": "custom", "prompt": args.prompt, "variants": args.variants})
    elif args.preset:
        jobs.append({"name": args.preset, "prompt": PRESETS[args.preset], "variants": args.variants})
    else:
        print("✗ Provide --preset, --prompt, or --all_presets")
        sys.exit(1)

    total_images = sum(j["variants"] for j in jobs)
    generated    = 0
    errors       = []

    for job in jobs:
        print(f"── {job['name']} ──")
        for v in range(1, job["variants"] + 1):
            filename = f"thumbnail_{job['name']}_v{v}.png"
            out_path = thumb_dir / filename
            print(f"  Variant {v}/{job['variants']}...", end=" ", flush=True)
            t = time.time()
            try:
                workflow  = build_workflow(
                    prompt     = f"{job['prompt']}, {STYLE_SUFFIX}",
                    negative   = NEGATIVE_PROMPT,
                    checkpoint = args.checkpoint,
                    steps      = args.steps,
                    cfg        = args.cfg,
                    seed       = args.seed,
                    width      = THUMB_WIDTH,
                    height     = THUMB_HEIGHT,
                )
                img_info  = queue_and_wait(workflow, args.api_url)
                img_bytes = download_image(img_info, args.api_url)
                out_path.write_bytes(img_bytes)
                generated += 1
                print(f"done in {elapsed(t)}  →  {filename}")
            except Exception as ex:
                print(f"FAILED: {ex}")
                errors.append(filename)
        print()

    print(f"{'='*55}")
    print(f"  Done!")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Generated  : {generated} / {total_images}")
    if errors:
        print(f"  Failed     : {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    print(f"  Output     : {thumb_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()