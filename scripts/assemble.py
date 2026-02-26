"""
assemble.py — Combine full audio with a looping animation into a YouTube-ready MP4
Usage: python3 scripts/assemble.py --video_name my_video --animation path/to/loop.mp4

Input:
  - generated/<video_name>/<video_name>_full.mp3
  - A short looping animation (.mp4 or .gif)

Output:
  - generated/<video_name>/<video_name>_full.mp4  ← YouTube-ready

YouTube recommended settings applied:
  - Resolution : 1920x1080 (1080p)
  - Video codec: H.264
  - Audio codec: AAC 192k
  - FPS        : 30
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path


# ─────────────────────────────────────────────
# Timer helper
# ─────────────────────────────────────────────

def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble video from audio + looping animation")
    parser.add_argument("--video_name",  required=True,
                        help="Name of the video folder inside generated/")
    parser.add_argument("--animation",   required=True,
                        help="Path to the looping animation (.mp4 or .gif)")
    parser.add_argument("--resolution",  default="1920x1080",
                        help="Output resolution (default: 1920x1080)")
    parser.add_argument("--fps",         default=30, type=int,
                        help="Output FPS (default: 30)")
    parser.add_argument("--audio_file",  default=None,
                        help="Override audio file path (default: auto-detected full.mp3)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Get audio duration via ffprobe
# ─────────────────────────────────────────────

def get_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return float(result.stdout.strip())


# ─────────────────────────────────────────────
# Format seconds nicely
# ─────────────────────────────────────────────

def fmt_duration(seconds):
    s = int(seconds)
    return f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s" if s >= 3600 else f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────
# Assemble video using ffmpeg
# ─────────────────────────────────────────────

def assemble_video(audio_path, animation_path, out_path, resolution, fps, duration):
    """
    Uses ffmpeg to:
    1. Loop the animation for the full audio duration
    2. Scale to target resolution
    3. Mux with audio
    4. Export as H.264/AAC MP4
    """
    width, height = resolution.split("x")

    cmd = [
        "ffmpeg", "-y",

        # Loop animation input
        "-stream_loop", "-1",
        "-i", str(animation_path),

        # Audio input
        "-i", str(audio_path),

        # Video filters: scale to resolution, pad if needed, set fps
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps}"
        ),

        # Stop at audio duration
        "-t", str(duration),

        # Video codec
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        # Audio codec
        "-c:a", "aac",
        "-b:a", "192k",

        # Shortest stream wins (safety net)
        "-shortest",

        str(out_path)
    ]

    print(f"   Running ffmpeg...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n✗ ffmpeg error:\n{result.stderr}")
        raise RuntimeError("ffmpeg assembly failed")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args        = parse_args()
    total_start = time.time()

    base_dir       = Path(__file__).parent.parent / "generated"
    video_dir      = base_dir / args.video_name
    animation_path = Path(args.animation)

    # ── Validate inputs ──
    if not video_dir.exists():
        print(f"✗ Video folder not found: {video_dir}")
        sys.exit(1)

    if not animation_path.exists():
        print(f"✗ Animation file not found: {animation_path}")
        sys.exit(1)

    if animation_path.suffix.lower() not in [".mp4", ".gif", ".webm", ".mov"]:
        print(f"✗ Unsupported animation format: {animation_path.suffix}")
        print(f"  Supported: .mp4, .gif, .webm, .mov")
        sys.exit(1)

    # ── Find audio file ──
    if args.audio_file:
        audio_path = Path(args.audio_file)
    else:
        audio_path = None
        for ext in ["mp3", "wav"]:
            candidate = video_dir / f"{args.video_name}_full.{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            print(f"✗ No full audio file found in {video_dir}")
            print(f"  Expected: {args.video_name}_full.mp3 or {args.video_name}_full.wav")
            print(f"  Run stitch.py first.")
            sys.exit(1)

    out_path = video_dir / f"{args.video_name}_full.mp4"

    # ── Print summary ──
    print(f"\n{'='*55}")
    print(f"  Video Assembly")
    print(f"  Video name : {args.video_name}")
    print(f"  Audio      : {audio_path.name}")
    print(f"  Animation  : {animation_path.name}")
    print(f"  Resolution : {args.resolution}")
    print(f"  FPS        : {args.fps}")
    print(f"{'='*55}\n")

    # ── Get audio duration ──
    print("Getting audio duration...")
    duration = get_duration(audio_path)
    print(f"✓ Audio duration: {fmt_duration(duration)}\n")

    # ── Assemble ──
    print("Assembling video...")
    t = time.time()
    assemble_video(audio_path, animation_path, out_path, args.resolution, args.fps, duration)
    print(f"✓ Video assembled in {elapsed(t)}\n")

    # ── File size ──
    size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"{'='*55}")
    print(f"  Assembly complete!")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Output     : {out_path}")
    print(f"  File size  : {size_mb:.1f} MB")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()