"""
assemble_sets.py — Assemble one video per set from a prompts file
Each # comment block = one set = one video, matched to a visual from a folder.

Usage:
  python3 scripts/assemble_sets.py \
    --video_name my_project \
    --visuals assets/visuals/ \
    --prompts prompts.txt

Visuals folder:
  Place one .mp4 per set, named anything.
  They are assigned to sets in alphabetical order.
  Example:
    visuals/
      01_bedroom.mp4    → Set 1
      02_cafe.mp4       → Set 2
      03_rooftop.mp4    → Set 3

Output:
  generated/<video_name>/sets/
    set_001_<set_name>.mp4
    set_002_<set_name>.mp4
    ...
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────

def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble one video per set")
    parser.add_argument("--video_name", required=True,
        help="Project name (must match generated/<video_name>/)")
    parser.add_argument("--visuals",    required=True,
        help="Folder containing one .mp4 per set (assigned alphabetically)")
    parser.add_argument("--prompts",    required=True,
        help="Prompts .txt file (used to identify sets and their names)")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps",        default=30, type=int)
    parser.add_argument("--fade",       default=3, type=int,
        help="Fade in duration in seconds (default: 3)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Parse sets from prompts file
# Returns list of dicts: {name, slugs[]}
# ─────────────────────────────────────────────

def parse_sets(prompts_path):
    sets        = []
    current_set = None

    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # New set header — extract name after "Lofi Set N –" or just use full comment
                label = line.lstrip("#").strip()
                # Try to extract name after dash
                if "–" in label:
                    name = label.split("–", 1)[1].strip()
                elif "-" in label:
                    name = label.split("-", 1)[1].strip()
                else:
                    name = label
                # Slugify
                slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                if current_set:
                    sets.append(current_set)
                current_set = {"name": name, "slug": slug, "filenames": []}

            elif "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3 and current_set:
                    current_set["filenames"].append(parts[2])

    if current_set:
        sets.append(current_set)

    return sets


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
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    return float(result.stdout.strip())


# ─────────────────────────────────────────────
# Assemble one set video
# ─────────────────────────────────────────────

def assemble_set(stitched_path, visual_path, out_path, resolution, fps, fade):
    width, height = resolution.split("x")
    duration      = get_duration(stitched_path)

    vf_chain = ",".join([
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        f"fps={fps}",
        f"fade=t=in:st=0:d={fade}"
    ])

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(visual_path),
        "-i", str(stitched_path),
        "-t", str(duration),
        "-vf", vf_chain,
        "-af", f"afade=t=in:st=0:d={fade}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args        = parse_args()
    total_start = time.time()

    base_dir    = Path("generated") / args.video_name
    visuals_dir = Path(args.visuals)
    sets_dir    = base_dir / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate ──
    if not base_dir.exists():
        print(f"✗ Project folder not found: {base_dir}")
        sys.exit(1)
    if not visuals_dir.exists():
        print(f"✗ Visuals folder not found: {visuals_dir}")
        sys.exit(1)

    # ── Parse sets from prompts ──
    sets = parse_sets(args.prompts)
    if not sets:
        print("✗ No sets found in prompts file")
        sys.exit(1)

    # ── Collect visuals (alphabetical) ──
    visuals = sorted([
        f for f in visuals_dir.iterdir()
        if f.suffix.lower() in [".mp4", ".gif", ".mov", ".webm"]
    ])

    if len(visuals) < len(sets):
        print(f"⚠  {len(sets)} sets found but only {len(visuals)} visual(s) in folder")
        print(f"   Visuals will be reused cyclically")

    print(f"\n{'='*55}")
    print(f"  Set Assembler")
    print(f"  Project    : {args.video_name}")
    print(f"  Sets found : {len(sets)}")
    print(f"  Visuals    : {len(visuals)}")
    print(f"  Output     : {sets_dir}")
    print(f"{'='*55}\n")

    # ── Find stitched audio per set ──
    # Look for part folders and their stitched files
    part_folders = sorted(base_dir.glob(f"{args.video_name}_part_*"))

    # Map each set to its stitched audio file
    # Sets are assigned to parts in order
    if len(part_folders) < len(sets):
        print(f"⚠  {len(sets)} sets but only {len(part_folders)} part folder(s) found")
        print(f"   Make sure stitch.py has been run first")

    errors = []

    for i, set_info in enumerate(sets):
        set_num     = i + 1
        visual_path = visuals[i % len(visuals)]  # cycle if fewer visuals than sets

        # Find stitched file for this set
        # Each set maps to one part folder in order
        if i < len(part_folders):
            part_dir      = part_folders[i]
            stitched_files = list(part_dir.glob("*_stitched.*"))
            if not stitched_files:
                print(f"  [{set_num:02d}/{len(sets)}] ✗ No stitched file in {part_dir.name} — skipping")
                errors.append(set_info["name"])
                continue
            stitched_path = stitched_files[0]
        else:
            print(f"  [{set_num:02d}/{len(sets)}] ✗ No part folder for set {set_num} — skipping")
            errors.append(set_info["name"])
            continue

        out_name = f"set_{set_num:03d}_{set_info['slug']}.mp4"
        out_path = sets_dir / out_name

        print(f"  [{set_num:02d}/{len(sets)}] {set_info['name']}")
        print(f"         Audio  : {stitched_path.name}")
        print(f"         Visual : {visual_path.name}")
        print(f"         Output : {out_name}")

        t = time.time()
        try:
            assemble_set(stitched_path, visual_path, out_path, args.resolution, args.fps, args.fade)
            size_mb = out_path.stat().st_size / (1024 ** 2)
            print(f"         ✓ done in {elapsed(t)}  |  {size_mb:.1f} MB\n")
        except Exception as ex:
            print(f"         ✗ FAILED: {ex}\n")
            errors.append(set_info["name"])

    # ── Summary ──
    print(f"{'='*55}")
    print(f"  Assembly complete!")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Videos     : {len(sets) - len(errors)} / {len(sets)}")
    if errors:
        print(f"  Failed     : {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    print(f"  Output dir : {sets_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()