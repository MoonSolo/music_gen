"""
stitch.py — Stitch generated clips into full tracks with crossfades
Usage: python3 scripts/stitch.py --video_name my_video [--crossfade 3] [--format mp3]

For each part folder inside generated/<video_name>/, it:
  1. Loads all .wav clips in order
  2. Normalizes volume across all clips
  3. Crossfades them together
  4. Exports a single audio file per part
  5. Exports a final combined file for the full video

Output structure:
  generated/
    <video_name>/
      <video_name>_part_001/
        clip_001.wav
        ...
        <video_name>_part_001_stitched.mp3   ← stitched track
      <video_name>_part_002/
        ...
        <video_name>_part_002_stitched.mp3
      <video_name>_full.mp3                  ← full 1h+ audio
"""

import argparse
import os
import sys
import time
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize


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
    parser = argparse.ArgumentParser(description="Stitch generated clips into full tracks")
    parser.add_argument("--video_name", required=True,
                        help="Name of the video folder inside generated/")
    parser.add_argument("--crossfade",  type=int, default=3,
                        help="Crossfade duration in seconds between clips (default: 3)")
    parser.add_argument("--format",     default="mp3", choices=["mp3", "wav"],
                        help="Output format (default: mp3)")
    parser.add_argument("--no_combine", action="store_true",
                        help="Skip creating the full combined file")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Load and normalize a single wav file
# ─────────────────────────────────────────────

def load_clip(path: Path) -> AudioSegment:
    audio = AudioSegment.from_wav(str(path))
    audio = normalize(audio)
    return audio


# ─────────────────────────────────────────────
# Stitch a list of AudioSegments with crossfade
# ─────────────────────────────────────────────

def stitch_clips(clips: list, crossfade_ms: int) -> AudioSegment:
    if not clips:
        raise ValueError("No clips to stitch")
    
    result = clips[0]
    for clip in clips[1:]:
        result = result.append(clip, crossfade=crossfade_ms)
    
    return result


# ─────────────────────────────────────────────
# Export audio segment
# ─────────────────────────────────────────────

def export_audio(audio: AudioSegment, out_path: Path, fmt: str):
    if fmt == "mp3":
        audio.export(str(out_path), format="mp3", bitrate="192k")
    else:
        audio.export(str(out_path), format="wav")


# ─────────────────────────────────────────────
# Format duration nicely
# ─────────────────────────────────────────────

def fmt_duration(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s" if s >= 3600 else f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args       = parse_args()
    total_start = time.time()
    crossfade_ms = args.crossfade * 1000

    base_dir  = Path(__file__).parent.parent / "generated"
    video_dir = base_dir / args.video_name

    if not video_dir.exists():
        print(f"✗ Video folder not found: {video_dir}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Stitch Pipeline")
    print(f"  Video     : {args.video_name}")
    print(f"  Crossfade : {args.crossfade}s")
    print(f"  Format    : {args.format}")
    print(f"{'='*55}\n")

    # ── Find all part folders in order ──
    part_dirs = sorted([
        d for d in video_dir.iterdir()
        if d.is_dir() and "_part_" in d.name
    ])

    if not part_dirs:
        print("✗ No part folders found.")
        sys.exit(1)

    print(f"Found {len(part_dirs)} part folder(s)\n")

    stitched_parts = []

    for part_dir in part_dirs:
        print(f"── {part_dir.name} ──")

        # Find all .wav clips in order, excluding already stitched files
        wav_files = sorted([
            f for f in part_dir.glob("*.wav")
            if "stitched" not in f.name
        ])

        if not wav_files:
            print(f"   ⚠ No .wav files found, skipping\n")
            continue

        print(f"   {len(wav_files)} clip(s) found")

        # Load and normalize clips
        clips = []
        for wav in wav_files:
            print(f"   Loading {wav.name}...")
            clips.append(load_clip(wav))

        # Stitch
        print(f"   Stitching with {args.crossfade}s crossfade...")
        t = time.time()
        stitched = stitch_clips(clips, crossfade_ms)
        print(f"   Stitched in {elapsed(t)}  |  duration: {fmt_duration(len(stitched))}")

        # Export part file
        out_filename = f"{part_dir.name}_stitched.{args.format}"
        out_path     = part_dir / out_filename
        print(f"   Exporting {out_filename}...")
        export_audio(stitched, out_path, args.format)
        print(f"   ✓ Saved → {out_path}\n")

        stitched_parts.append(stitched)

    if not stitched_parts:
        print("✗ No parts were stitched. Exiting.")
        sys.exit(1)

    # ── Combine all parts into full video audio ──
    if not args.no_combine:
        print(f"── Combining all parts into full audio ──")
        t = time.time()
        full_audio = stitch_clips(stitched_parts, crossfade_ms)
        print(f"   Combined in {elapsed(t)}  |  total duration: {fmt_duration(len(full_audio))}")

        full_out = video_dir / f"{args.video_name}_full.{args.format}"
        print(f"   Exporting {full_out.name}...")
        export_audio(full_audio, full_out, args.format)
        print(f"   ✓ Saved → {full_out}\n")

    # ── Summary ──
    print(f"{'='*55}")
    print(f"  Stitching complete!")
    print(f"  Total time : {elapsed(total_start)}")
    if not args.no_combine:
        print(f"  Full audio : {args.video_name}_full.{args.format}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()