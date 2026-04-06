"""
generate.py — Batch music generation pipeline
Usage: python3 generate.py --model [small|medium|large] --prompts prompts.txt --video_name my_video

Folder structure:
  generated/
    <video_name>/
      <video_name>_part_001/   ← one folder per "track" (2–6 min)
        clip_001.wav
        clip_002.wav
        ...
      <video_name>_part_002/
        ...
"""

import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MusicGen batch pipeline")
    parser.add_argument("--model",      required=True, choices=["small", "medium", "large"],
                        help="MusicGen model to use")
    parser.add_argument("--prompts",    required=True,
                        help="Path to prompts .txt file")
    parser.add_argument("--video_name", required=True,
                        help="Name for this video/output folder")
    parser.add_argument("--device",     default=None,
                        help="Force device: 'cpu' or 'cuda'. Auto-detected if not set.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Prompt file parser
# ─────────────────────────────────────────────

def parse_prompt_file(filepath):
    """
    Format per line:
        prompt text | duration_seconds | output_filename
    Lines starting with # or blank are skipped.
    """
    entries = []
    errors  = []

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                errors.append(f"Line {line_num}: expected 3 fields, got {len(parts)}: '{line}'")
                continue

            prompt, duration_str, filename = parts

            if not prompt:
                errors.append(f"Line {line_num}: empty prompt")
                continue

            try:
                duration = int(duration_str)
                if not (5 <= duration <= 360):
                    raise ValueError
            except ValueError:
                errors.append(f"Line {line_num}: duration must be integer 5–360, got '{duration_str}'")
                continue

            if not filename:
                errors.append(f"Line {line_num}: empty filename")
                continue

            entries.append({
                "prompt":    prompt,
                "duration":  duration,
                "filename":  filename,
                "line_num":  line_num,
            })

    return entries, errors


# ─────────────────────────────────────────────
# Folder helpers
# ─────────────────────────────────────────────

def make_output_dirs(base_dir: Path, video_name: str, num_parts: int):
    """
    Creates:
      generated/<video_name>/<video_name>_part_001/
      generated/<video_name>/<video_name>_part_002/
      ...
    Returns list of part folder paths.
    """
    video_dir = base_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    part_dirs = []
    for i in range(1, num_parts + 1):
        part_dir = video_dir / f"{video_name}_part_{i:03d}"
        part_dir.mkdir(exist_ok=True)
        part_dirs.append(part_dir)

    return video_dir, part_dirs


# ─────────────────────────────────────────────
# Progress log (so we can resume)
# ─────────────────────────────────────────────

def load_progress(progress_file: Path):
    if progress_file.exists():
        with open(progress_file) as f:
            return set(json.load(f))
    return set()

def save_progress(progress_file: Path, completed: set):
    with open(progress_file, "w") as f:
        json.dump(list(completed), f)


# ─────────────────────────────────────────────
# Timer helper
# ─────────────────────────────────────────────

def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────
# Clip grouping: split entries into parts (2–6 min each)
# targeting ~1–2h total
# ─────────────────────────────────────────────

def group_into_parts(entries, min_part_sec=120, max_part_sec=360):
    """
    Groups consecutive entries into parts where each part's
    total duration is between min_part_sec and max_part_sec.
    """
    parts     = []
    current   = []
    current_t = 0

    for entry in entries:
        d = entry["duration"]
        # If adding this clip would exceed max, start a new part
        if current and (current_t + d > max_part_sec):
            parts.append(current)
            current   = []
            current_t = 0
        current.append(entry)
        current_t += d

    if current:
        parts.append(current)

    return parts


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    total_start = time.time()

    # ── Device selection ──
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"  MusicGen Batch Pipeline")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"  Prompts: {args.prompts}")
    print(f"  Video  : {args.video_name}")
    print(f"{'='*55}\n")

    # ── Parse prompts ──
    print("Parsing prompt file...")
    entries, errors = parse_prompt_file(args.prompts)

    if errors:
        print(f"\n⚠  {len(errors)} error(s) in prompt file:")
        for e in errors:
            print(f"   - {e}")

    if not entries:
        print("No valid entries found. Exiting.")
        sys.exit(1)

    total_duration = sum(e["duration"] for e in entries)
    print(f"✓  {len(entries)} clips loaded  |  total audio: {total_duration//60}m {total_duration%60}s\n")

    # ── Group into parts ──
    parts = group_into_parts(entries)
    print(f"✓  Grouped into {len(parts)} part(s)\n")
    for i, part in enumerate(parts, 1):
        part_dur = sum(e["duration"] for e in part)
        print(f"   Part {i:03d}: {len(part)} clips  |  {part_dur//60}m {part_dur%60}s")
    print()

    # ── Create output folders ──
    base_dir = Path(__file__).parent.parent / "generated"
    video_dir, part_dirs = make_output_dirs(base_dir, args.video_name, len(parts))
    print(f"✓  Output root: {video_dir}\n")

    # ── Progress file (for resuming) ──
    progress_file = video_dir / "progress.json"
    completed     = load_progress(progress_file)
    if completed:
        print(f"↺  Resuming — {len(completed)} clip(s) already done\n")

    # ── Load model ──
    print(f"Loading MusicGen '{args.model}' on {device}...")
    t = time.time()
    model = MusicGen.get_pretrained(args.model, device=device)
    print(f"✓  Model loaded in {elapsed(t)}\n")

    # ── Generate clips ──
    total_clips  = len(entries)
    done_count   = len(completed)
    fail_log     = []

    for part_idx, (part, part_dir) in enumerate(zip(parts, part_dirs), 1):
        print(f"── Part {part_idx:03d} / {len(parts):03d}  ({part_dir.name}) ──")

        for clip_idx, entry in enumerate(part, 1):
            clip_id = entry["filename"]

            # Skip already completed
            if clip_id in completed:
                print(f"   [skip] {clip_id}")
                continue

            out_path = part_dir / clip_id
            print(f"   [{done_count+1:03d}/{total_clips}] {clip_id}  |  {entry['duration']}s  |  {entry['prompt'][:60]}")

            try:
                t = time.time()
                model.set_generation_params(duration=entry["duration"])
                wav = model.generate([entry["prompt"]])
                audio_write(
                    str(out_path),
                    wav[0].cpu(),
                    model.sample_rate,
                    strategy="loudness"
                )
                gen_time = elapsed(t)
                done_count += 1
                completed.add(clip_id)
                save_progress(progress_file, completed)
                print(f"         ✓ done in {gen_time}  →  {out_path.name}.wav")

            except Exception as ex:
                print(f"         ✗ FAILED: {ex}")
                fail_log.append({"clip": clip_id, "error": str(ex)})

        print()

    # ── Summary ──
    print(f"{'='*55}")
    print(f"  Generation complete!")
    print(f"  Total time   : {elapsed(total_start)}")
    print(f"  Clips done   : {done_count} / {total_clips}")
    if fail_log:
        print(f"  Failed clips : {len(fail_log)}")
        for f in fail_log:
            print(f"    - {f['clip']}: {f['error']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()