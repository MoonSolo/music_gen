"""
run.py — Master automation script
Runs the full pipeline: generate → stitch → assemble

Usage:
  python3 scripts/run.py --model large --prompts prompts.txt --video_name my_video --animation assets/loop.mp4

Optional flags:
  --device cpu/cuda       Force device (default: auto)
  --crossfade 3           Crossfade duration in seconds (default: 3)
  --resolution 1920x1080  Output resolution (default: 1920x1080)
  --fps 30                Output FPS (default: 30)
  --format mp3            Audio format (default: mp3)
  --skip_generate         Skip generation, go straight to stitch+assemble
  --skip_stitch           Skip stitching, go straight to assemble
  --skip_assemble         Skip video assembly (audio pipeline only)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────
# Timer helper
# ─────────────────────────────────────────────

def elapsed(start):
    s = int(time.time() - start)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}h {m}m {sec}s" if h > 0 else f"{m}m {sec}s"


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MusicGen full pipeline runner")

    # Required
    parser.add_argument("--model",       required=True, choices=["small", "medium", "large"],
                        help="MusicGen model to use")
    parser.add_argument("--prompts",     required=True,
                        help="Path to prompts .txt file")
    parser.add_argument("--video_name",  required=True,
                        help="Name for this video")

    # Optional — generation
    parser.add_argument("--device",      default=None,
                        help="Force device: cpu or cuda (default: auto)")

    # Optional — stitching
    parser.add_argument("--crossfade",   default=3, type=int,
                        help="Crossfade duration in seconds (default: 3)")
    parser.add_argument("--format",      default="mp3", choices=["mp3", "wav"],
                        help="Stitched audio format (default: mp3)")

    # Optional — assembly
    parser.add_argument("--animation",   default=None,
                        help="Path to looping animation for video assembly")
    parser.add_argument("--resolution",  default="1920x1080",
                        help="Output video resolution (default: 1920x1080)")
    parser.add_argument("--fps",         default=30, type=int,
                        help="Output video FPS (default: 30)")

    # Skip flags
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip generation step")
    parser.add_argument("--skip_stitch",   action="store_true",
                        help="Skip stitching step")
    parser.add_argument("--skip_assemble", action="store_true",
                        help="Skip video assembly step")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Run a step as subprocess
# ─────────────────────────────────────────────

def run_step(label, cmd):
    print(f"\n{'='*55}")
    print(f"  STEP: {label}")
    print(f"{'='*55}\n")

    t = time.time()
    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\n✗ Step '{label}' failed with exit code {result.returncode}")
        print(f"  Aborting pipeline.")
        sys.exit(result.returncode)

    print(f"\n✓ Step '{label}' completed in {elapsed(t)}")
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args        = parse_args()
    total_start = time.time()
    scripts_dir = Path(__file__).parent

    print(f"\n{'='*55}")
    print(f"  MusicGen Full Pipeline")
    print(f"  Model      : {args.model}")
    print(f"  Video name : {args.video_name}")
    print(f"  Prompts    : {args.prompts}")
    if args.animation:
        print(f"  Animation  : {args.animation}")
    print(f"{'='*55}")

    steps_run = []

    # ── Step 1: Generate ──
    if not args.skip_generate:
        cmd = [
            sys.executable, str(scripts_dir / "generate.py"),
            "--model",      args.model,
            "--prompts",    args.prompts,
            "--video_name", args.video_name,
        ]
        if args.device:
            cmd += ["--device", args.device]

        run_step("Generate clips", cmd)
        steps_run.append("generate")
    else:
        print("\n↺  Skipping generation step")

    # ── Step 2: Stitch ──
    if not args.skip_stitch:
        cmd = [
            sys.executable, str(scripts_dir / "stitch.py"),
            "--video_name", args.video_name,
            "--crossfade",  str(args.crossfade),
            "--format",     args.format,
        ]

        run_step("Stitch clips", cmd)
        steps_run.append("stitch")
    else:
        print("\n↺  Skipping stitch step")

    # ── Step 3: Assemble ──
    if not args.skip_assemble:
        if not args.animation:
            print("\n⚠  --animation not provided, skipping video assembly.")
            print("   Run assemble.py manually when your animation is ready:")
            print(f"   python3 scripts/assemble.py --video_name {args.video_name} --animation <path>")
        else:
            cmd = [
                sys.executable, str(scripts_dir / "assemble.py"),
                "--video_name", args.video_name,
                "--animation",  args.animation,
                "--resolution", args.resolution,
                "--fps",        str(args.fps),
            ]

            run_step("Assemble video", cmd)
            steps_run.append("assemble")
    else:
        print("\n↺  Skipping assembly step")

    # ── Final summary ──
    print(f"\n{'='*55}")
    print(f"  Pipeline complete!")
    print(f"  Steps run  : {' → '.join(steps_run) if steps_run else 'none'}")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Output     : generated/{args.video_name}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()