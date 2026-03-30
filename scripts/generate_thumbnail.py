"""
generate_thumbnail.py — Generate YouTube thumbnails via ComfyUI API
Requires ComfyUI running: cd ~/ComfyUI && source venv/bin/activate && python main.py --listen 127.0.0.1 --port 7860

Usage:
  python3 scripts/generate_thumbnail.py --video_name lofi_night --preset coding_3am
  python3 scripts/generate_thumbnail.py --video_name lofi_night --preset coding_3am --variants 3
  python3 scripts/generate_thumbnail.py --video_name lofi_night --prompt "anime girl at window, neon city"
  python3 scripts/generate_thumbnail.py --video_name lofi_night --all_presets

Presets: coding_3am, insomnia_1am, night_drive, overthinking_2am,
         rooftop_night, convenience_store, window_rain, desk_glow
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
    "anime style, Studio Ghibli inspired, vaporwave aesthetic, "
    "cyberpunk night, neon lighting, soft glow, highly detailed, "
    "cinematic composition, late night atmosphere, 4k, masterpiece, "
    "best quality, illustration, lofi aesthetic"
)

NEGATIVE_PROMPT = (
    "nsfw, realistic, photograph, 3d render, ugly, deformed, "
    "blurry, low quality, bad anatomy, watermark, signature, "
    "text, logo, harsh lighting, oversaturated, cartoon, chibi, "
    "daytime, bright sunlight, cheerful, busy background"
)

PRESETS = {
    "coding_3am": (
        "anime girl at glowing computer desk, 3am dark bedroom, "
        "multiple monitors casting blue light, cyberpunk city visible through rain window, "
        "headphones on, focused expression, neon purple and pink tones, "
        "vaporwave aesthetic, lonely but cozy atmosphere"
    ),
    "insomnia_1am": (
        "anime girl lying in bed staring at ceiling, 1am digital clock, "
        "dark room with faint neon light from window, city glow outside, "
        "melancholic expression, soft purple moonlight, vaporwave color palette, "
        "barely visible silhouette, introspective mood"
    ),
    "night_drive": (
        "anime girl in car passenger seat at night, rain on window, "
        "neon city lights streaking past, reflections on wet glass, "
        "soft expression looking outside, cyberpunk cityscape, "
        "pink and blue neon tones, vaporwave night aesthetic"
    ),
    "overthinking_2am": (
        "anime girl sitting on bedroom floor against bed, 2am, "
        "knees drawn up, phone screen glow in dark room, "
        "neon signs visible through window, vaporwave purple haze, "
        "emotional introspective mood, alone but relatable"
    ),
    "rooftop_night": (
        "anime girl sitting on rooftop at night, cyberpunk city below, "
        "neon lights reflecting off wet rooftop surface, "
        "looking at city skyline, headphones around neck, "
        "vaporwave pink and purple sky, solitary peaceful mood"
    ),
    "convenience_store": (
        "anime girl inside convenience store at 3am, alone, "
        "bright white fluorescent light contrasting dark street outside, "
        "neon reflections on wet pavement, vaporwave aesthetic, "
        "soft melancholic expression, urban loneliness, relatable moment"
    ),
    "window_rain": (
        "anime girl sitting at window watching rain at night, "
        "warm indoor light behind her, cold blue neon city outside, "
        "condensation on glass, soft silhouette, vaporwave color split, "
        "introspective mood, barely visible face, emotionally resonant"
    ),
    "desk_glow": (
        "anime girl at desk late at night, single lamp glow, "
        "open notebook and coffee cup, dark room, cyberpunk city outside window, "
        "neon pink and purple light bleeding in, vaporwave aesthetic, "
        "tired but persisting expression, relatable late night study mood"
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
    parser.add_argument("--checkpoint",  default="Counterfeit-V3.safetensors",
        help="Main model checkpoint filename")
    parser.add_argument("--vae",         default="kl-f8-anime2.ckpt",
        help="VAE filename (default: kl-f8-anime2.vae.pt). Use 'none' to skip.")
    parser.add_argument("--api_url",     default=COMFY_URL)
    return parser.parse_args()


def check_api(api_url):
    try:
        r = requests.get(f"{api_url}/system_stats", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def build_workflow(prompt, negative, checkpoint, vae, steps, cfg, seed, width, height):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    use_vae = vae.lower() != "none"

    # Node layout:
    # 1 = CheckpointLoaderSimple
    # 2 = VAELoader (optional)
    # 3 = CLIPTextEncode (positive)
    # 4 = CLIPTextEncode (negative)
    # 5 = EmptyLatentImage
    # 6 = KSampler
    # 7 = VAEDecode
    # 8 = SaveImage

    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["3", 0],
                "negative":     ["4", 0],
                "latent_image": ["5", 0],
                "seed":         seed,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": "dpm_2_ancestral",
                "scheduler":    "karras",
                "denoise":      1.0
            }
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images":            ["7", 0],
                "filename_prefix":   "thumbnail"
            }
        }
    }

    if use_vae:
        # Load external VAE and use it for decoding
        workflow["2"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae}
        }
        workflow["7"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae":     ["2", 0]   # external VAE
            }
        }
    else:
        # Use VAE baked into the checkpoint
        workflow["7"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae":     ["1", 2]   # checkpoint VAE
            }
        }

    return {"prompt": workflow, "client_id": CLIENT_ID}


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

    use_vae = args.vae.lower() != "none"

    print(f"\n{'='*55}")
    print(f"  Thumbnail Generator (ComfyUI)")
    print(f"  Video      : {args.video_name}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  VAE        : {args.vae if use_vae else 'built-in (checkpoint)'}")
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
                    vae        = args.vae,
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