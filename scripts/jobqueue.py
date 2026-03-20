"""
queue.py — Queue multiple prompt files for large batch generation
Runs the full pipeline (generate + stitch + assemble) for each job in sequence.

Usage:
  python3 scripts/queue.py --jobs queue.txt

Queue file format (queue.txt):
  Each line defines one job:
  prompts_file | video_name | animation_file | model

  Example:
    prompts/lofi_vol1.txt | lofi_vol1 | assets/bedroom.mp4 | small
    prompts/lofi_vol2.txt | lofi_vol2 | assets/cafe.mp4    | small
    prompts/ambient.txt   | ambient_1 | assets/forest.mp4  | small

  Lines starting with # are ignored.
  Leave animation empty to skip assembly:
    prompts/lofi_vol3.txt | lofi_vol3 | | small

Options:
  --device cuda/cpu     Force device for all jobs (default: auto)
  --skip_generate       Skip generation for all jobs
  --skip_stitch         Skip stitching for all jobs
  --skip_assemble       Skip assembly for all jobs
  --start_from N        Start from job N (1-indexed), skip previous
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
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}h {m}m {sec}s" if h > 0 else f"{m}m {sec}s"


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Queue multiple pipeline jobs")
    parser.add_argument("--jobs",         required=True,
        help="Path to queue file")
    parser.add_argument("--device",       default=None,
        help="Force device: cpu or cuda (default: auto)")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_stitch",   action="store_true")
    parser.add_argument("--skip_assemble", action="store_true")
    parser.add_argument("--start_from",   default=1, type=int,
        help="Start from job N (1-indexed). Skip all previous jobs.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Parse queue file
# ─────────────────────────────────────────────

def parse_queue(filepath):
    jobs   = []
    errors = []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                errors.append(f"Line {line_num}: expected at least 3 fields, got {len(parts)}")
                continue

            prompts   = parts[0]
            video     = parts[1]
            animation = parts[2] if len(parts) > 2 else ""
            model     = parts[3] if len(parts) > 3 else "small"

            if not prompts or not video:
                errors.append(f"Line {line_num}: prompts and video_name are required")
                continue

            if not Path(prompts).exists():
                errors.append(f"Line {line_num}: prompts file not found: {prompts}")
                continue

            if animation and not Path(animation).exists():
                errors.append(f"Line {line_num}: animation file not found: {animation}")
                continue

            if model not in ["small", "medium", "large"]:
                errors.append(f"Line {line_num}: invalid model '{model}', must be small/medium/large")
                continue

            jobs.append({
                "prompts":   prompts,
                "video":     video,
                "animation": animation,
                "model":     model,
                "line_num":  line_num,
            })

    return jobs, errors


# ─────────────────────────────────────────────
# Run one job
# ─────────────────────────────────────────────

def run_job(job, args, job_num, total_jobs):
    scripts_dir = Path(__file__).parent
    job_start   = time.time()
    failed      = []

    print(f"\n{'='*55}")
    print(f"  JOB {job_num}/{total_jobs} — {job['video']}")
    print(f"  Prompts   : {job['prompts']}")
    print(f"  Model     : {job['model']}")
    print(f"  Animation : {job['animation'] or '(none)'}")
    print(f"{'='*55}\n")

    # ── Generate ──
    if not args.skip_generate:
        cmd = [
            sys.executable, str(scripts_dir / "generate.py"),
            "--model",      job["model"],
            "--prompts",    job["prompts"],
            "--video_name", job["video"],
        ]
        if args.device:
            cmd += ["--device", args.device]

        print(f"  → Generating clips...")
        t = time.time()
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"  ✗ Generation failed for job {job_num}")
            failed.append("generate")
            return False, elapsed(job_start)
        print(f"  ✓ Generation done in {elapsed(t)}")
    else:
        print(f"  ↺ Skipping generation")

    # ── Stitch ──
    if not args.skip_stitch:
        cmd = [
            sys.executable, str(scripts_dir / "stitch.py"),
            "--video_name", job["video"],
        ]

        print(f"  → Stitching clips...")
        t = time.time()
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"  ✗ Stitching failed for job {job_num}")
            return False, elapsed(job_start)
        print(f"  ✓ Stitching done in {elapsed(t)}")
    else:
        print(f"  ↺ Skipping stitching")

    # ── Assemble ──
    if not args.skip_assemble:
        if not job["animation"]:
            print(f"  ↺ No animation provided — skipping assembly")
        else:
            cmd = [
                sys.executable, str(scripts_dir / "assemble.py"),
                "--video_name", job["video"],
                "--animation",  job["animation"],
            ]

            print(f"  → Assembling video...")
            t = time.time()
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print(f"  ✗ Assembly failed for job {job_num}")
                return False, elapsed(job_start)
            print(f"  ✓ Assembly done in {elapsed(t)}")
    else:
        print(f"  ↺ Skipping assembly")

    return True, elapsed(job_start)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args        = parse_args()
    total_start = time.time()

    # ── Parse queue ──
    print(f"\nParsing queue file: {args.jobs}")
    jobs, errors = parse_queue(args.jobs)

    if errors:
        print(f"\n⚠  {len(errors)} error(s) in queue file:")
        for e in errors:
            print(f"   - {e}")

    if not jobs:
        print("✗ No valid jobs found. Exiting.")
        sys.exit(1)

    print(f"✓ {len(jobs)} job(s) loaded\n")

    if args.start_from > 1:
        print(f"↺ Starting from job {args.start_from} — skipping {args.start_from - 1} job(s)\n")

    # ── Print job list ──
    print(f"{'='*55}")
    print(f"  Queue Summary")
    print(f"{'='*55}")
    for i, job in enumerate(jobs, 1):
        status = "→" if i >= args.start_from else "↺"
        print(f"  {status} [{i:02d}] {job['video']:<25} model={job['model']}")
    print(f"{'='*55}\n")

    # ── Run jobs ──
    results = []

    for i, job in enumerate(jobs, 1):
        if i < args.start_from:
            print(f"  Skipping job {i}/{len(jobs)}: {job['video']}")
            results.append({"job": job, "success": None, "time": "-"})
            continue

        success, job_time = run_job(job, args, i, len(jobs))
        results.append({"job": job, "success": success, "time": job_time})

        status = "✓" if success else "✗"
        print(f"\n  {status} Job {i} ({job['video']}) finished in {job_time}\n")

    # ── Final summary ──
    done    = [r for r in results if r["success"] is True]
    failed  = [r for r in results if r["success"] is False]
    skipped = [r for r in results if r["success"] is None]

    print(f"\n{'='*55}")
    print(f"  Queue complete!")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Completed  : {len(done)}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Skipped    : {len(skipped)}")

    if failed:
        print(f"\n  Failed jobs:")
        for r in failed:
            print(f"    - {r['job']['video']}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()