"""
whitenoise.py — Customisable white noise generator
Generates multiple variants of any noise type, lets you pick the best,
then extends it to any target duration.

Usage:
  python3 scripts/whitenoise.py --type brown --variants 5 --preview 10 --duration 36000 --output sleep_brown

Noise types:
  white     — Full spectrum, bright, harsh
  pink      — Softer, natural, balanced
  brown     — Deep, rumbling, like a fan or distant thunder
  blue      — High frequency emphasis, sharp
  violet    — Even higher frequency, like a hiss
  rain      — Synthesized rainfall texture
  ocean     — Rhythmic wave pattern
  fan       — Mechanical fan/AC hum
  womb      — Deep low rumble, for babies
  binaural  — Binaural beats embedded in pink noise (requires --freq_left and --freq_right)
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.io import wavfile


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
    parser = argparse.ArgumentParser(
        description="Customisable white noise generator",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Core
    parser.add_argument("--type", required=True,
        choices=["white", "pink", "brown", "blue", "violet", "rain", "ocean", "fan", "womb", "binaural"],
        help="Type of noise to generate")
    parser.add_argument("--output", required=True,
        help="Base output name (e.g. 'sleep_brown')")

    # Generation
    parser.add_argument("--variants", type=int, default=3,
        help="Number of variants to generate for preview (default: 3)")
    parser.add_argument("--preview", type=int, default=30,
        help="Duration of each preview variant in seconds (default: 30)")
    parser.add_argument("--duration", type=int, default=36000,
        help="Final output duration in seconds (default: 36000 = 10h)")
    parser.add_argument("--samplerate", type=int, default=44100,
        help="Sample rate in Hz (default: 44100)")

    # Audio shaping
    parser.add_argument("--volume", type=float, default=0.6,
        help="Output volume 0.0–1.0 (default: 0.6)")
    parser.add_argument("--fade_in", type=int, default=3,
        help="Fade in duration in seconds (default: 3)")
    parser.add_argument("--fade_out", type=int, default=3,
        help="Fade out duration in seconds (default: 3)")
    parser.add_argument("--stereo", action="store_true", default=True,
        help="Generate stereo output (default: True)")
    parser.add_argument("--mono", action="store_true",
        help="Force mono output")

    # Noise-specific options
    parser.add_argument("--rain_intensity", type=float, default=0.5,
        help="Rain intensity 0.0–1.0 (default: 0.5)")
    parser.add_argument("--rain_rumble", type=float, default=0.3,
        help="Thunder/rumble amount in rain 0.0–1.0 (default: 0.3)")
    parser.add_argument("--ocean_wave_speed", type=float, default=0.08,
        help="Ocean wave speed in Hz (default: 0.08 = ~12s per wave)")
    parser.add_argument("--ocean_intensity", type=float, default=0.5,
        help="Ocean wave intensity 0.0–1.0 (default: 0.5)")
    parser.add_argument("--fan_speed", type=float, default=50.0,
        help="Fan blade frequency in Hz (default: 50)")
    parser.add_argument("--fan_harmonics", type=int, default=6,
        help="Number of fan harmonic layers (default: 6)")
    parser.add_argument("--womb_depth", type=float, default=0.7,
        help="Womb rumble depth 0.0–1.0 (default: 0.7)")
    parser.add_argument("--freq_left", type=float, default=40.0,
        help="Binaural left ear frequency in Hz (default: 40)")
    parser.add_argument("--freq_right", type=float, default=50.0,
        help="Binaural right ear frequency in Hz (default: 50)")
    parser.add_argument("--lowcut", type=float, default=None,
        help="Low cut filter in Hz (removes frequencies below this)")
    parser.add_argument("--highcut", type=float, default=None,
        help="High cut filter in Hz (removes frequencies above this)")

    # Workflow
    parser.add_argument("--skip_preview", action="store_true",
        help="Skip variant generation, go straight to full duration output")
    parser.add_argument("--pick", type=int, default=None,
        help="Auto-select this variant number (skips manual prompt)")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Core noise generators
# ─────────────────────────────────────────────

def gen_white(n_samples):
    return np.random.normal(0, 1, n_samples)


def gen_pink(n_samples):
    """Voss-McCartney pink noise algorithm."""
    cols  = 16
    array = np.zeros((n_samples, cols))
    array[0] = np.random.random(cols)
    for i in range(1, n_samples):
        j = (~(i & -i)).bit_length() % cols
        array[i] = array[i - 1]
        array[i, j] = np.random.random()
    return (array.sum(axis=1) - cols / 2) / cols


def gen_brown(n_samples):
    """Brown (Brownian) noise via cumulative sum of white noise."""
    white  = np.random.normal(0, 1, n_samples)
    brown  = np.cumsum(white)
    brown -= np.mean(brown)
    # Normalize
    max_val = np.max(np.abs(brown))
    if max_val > 0:
        brown /= max_val
    return brown


def gen_blue(n_samples, sr):
    """Blue noise — differentiated white noise."""
    white = np.random.normal(0, 1, n_samples)
    b, a  = signal.butter(1, 0.01, btype='high', fs=sr)
    return signal.lfilter(b, a, white)


def gen_violet(n_samples, sr):
    """Violet noise — double differentiated."""
    white = np.random.normal(0, 1, n_samples)
    b, a  = signal.butter(2, 0.01, btype='high', fs=sr)
    return signal.lfilter(b, a, white)


def gen_rain(n_samples, sr, intensity=0.5, rumble=0.3):
    """
    Synthesized rain:
    - High freq filtered white noise for droplet texture
    - Pink noise base for air/atmosphere
    - Low freq brown noise for distant thunder rumble
    """
    # Droplet layer — highpass filtered white
    white  = np.random.normal(0, 1, n_samples)
    b, a   = signal.butter(4, 2000 / (sr / 2), btype='high')
    drops  = signal.lfilter(b, a, white) * intensity

    # Atmosphere layer — bandpass pink noise
    pink   = gen_pink(n_samples)
    b, a   = signal.butter(2, [200 / (sr / 2), 4000 / (sr / 2)], btype='band')
    air    = signal.lfilter(b, a, pink) * 0.4

    # Rumble layer — lowpass brown
    brown  = gen_brown(n_samples)
    b, a   = signal.butter(4, 120 / (sr / 2), btype='low')
    thunder = signal.lfilter(b, a, brown) * rumble

    return drops + air + thunder


def gen_ocean(n_samples, sr, wave_speed=0.08, intensity=0.5):
    """
    Synthesized ocean:
    - Pink noise base shaped by slow sinusoidal wave envelope
    - Low rumble layer
    """
    t      = np.linspace(0, n_samples / sr, n_samples)
    pink   = gen_pink(n_samples)

    # Wave envelope — slow sine modulation
    envelope = (1 + np.sin(2 * np.pi * wave_speed * t)) / 2
    envelope = np.power(envelope, 1.5)  # sharper wave crests

    # Bandpass for oceanic texture
    b, a   = signal.butter(3, [80 / (sr / 2), 6000 / (sr / 2)], btype='band')
    shaped = signal.lfilter(b, a, pink) * envelope * intensity

    # Deep rumble
    brown  = gen_brown(n_samples)
    b, a   = signal.butter(4, 100 / (sr / 2), btype='low')
    rumble = signal.lfilter(b, a, brown) * 0.3

    return shaped + rumble


def gen_fan(n_samples, sr, speed=50.0, harmonics=6):
    """
    Mechanical fan noise:
    - Fundamental + harmonics of blade frequency
    - Pink noise base for air movement
    - Slight random wobble for realism
    """
    t      = np.linspace(0, n_samples / sr, n_samples)
    result = np.zeros(n_samples)

    for h in range(1, harmonics + 1):
        freq    = speed * h
        if freq < sr / 2:
            amp   = 1.0 / (h ** 1.5)
            wobble = 1 + 0.002 * np.random.normal(0, 1, n_samples).cumsum() / sr
            wobble = np.clip(wobble, 0.99, 1.01)
            result += amp * np.sin(2 * np.pi * freq * t * wobble)

    # Air movement — pink noise
    pink  = gen_pink(n_samples)
    b, a  = signal.butter(3, [80 / (sr / 2), 3000 / (sr / 2)], btype='band')
    air   = signal.lfilter(b, a, pink) * 0.4

    return result * 0.5 + air


def gen_womb(n_samples, sr, depth=0.7):
    """
    Womb sound:
    - Deep brown noise base
    - Rhythmic low pulse (heartbeat-like, ~60 BPM)
    - Muffled texture via aggressive lowpass
    """
    brown  = gen_brown(n_samples)
    t      = np.linspace(0, n_samples / sr, n_samples)

    # Heartbeat-like pulse ~60 BPM
    pulse  = np.sin(2 * np.pi * 1.0 * t) ** 2
    pulse  = np.where(pulse > 0.8, pulse, 0) * depth

    # Aggressive lowpass — muffled in-utero effect
    b, a   = signal.butter(6, 300 / (sr / 2), btype='low')
    muffled = signal.lfilter(b, a, brown + pulse)

    return muffled


def gen_binaural(n_samples, sr, freq_left=40.0, freq_right=50.0):
    """
    Binaural beats embedded in pink noise.
    Left and right channels have slightly different pure tones.
    Brain perceives the difference frequency as a beat.
    REQUIRES stereo output.
    """
    t     = np.linspace(0, n_samples / sr, n_samples)
    pink  = gen_pink(n_samples)

    # Carrier frequency (audible tone, mid range)
    carrier = 200.0
    left_tone  = 0.15 * np.sin(2 * np.pi * (carrier + freq_left)  * t)
    right_tone = 0.15 * np.sin(2 * np.pi * (carrier + freq_right) * t)

    # Pink noise base
    b, a  = signal.butter(2, [100 / (sr / 2), 8000 / (sr / 2)], btype='band')
    base  = signal.lfilter(b, a, pink) * 0.7

    left  = base + left_tone
    right = base + right_tone

    return np.stack([left, right], axis=1)


# ─────────────────────────────────────────────
# Dispatch generator
# ─────────────────────────────────────────────

def generate_noise(noise_type, n_samples, sr, args):
    if noise_type == "white":
        mono = gen_white(n_samples)
    elif noise_type == "pink":
        mono = gen_pink(n_samples)
    elif noise_type == "brown":
        mono = gen_brown(n_samples)
    elif noise_type == "blue":
        mono = gen_blue(n_samples, sr)
    elif noise_type == "violet":
        mono = gen_violet(n_samples, sr)
    elif noise_type == "rain":
        mono = gen_rain(n_samples, sr, args.rain_intensity, args.rain_rumble)
    elif noise_type == "ocean":
        mono = gen_ocean(n_samples, sr, args.ocean_wave_speed, args.ocean_intensity)
    elif noise_type == "fan":
        mono = gen_fan(n_samples, sr, args.fan_speed, args.fan_harmonics)
    elif noise_type == "womb":
        mono = gen_womb(n_samples, sr, args.womb_depth)
    elif noise_type == "binaural":
        # Binaural returns stereo directly
        return gen_binaural(n_samples, sr, args.freq_left, args.freq_right)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return mono


# ─────────────────────────────────────────────
# Audio post-processing
# ─────────────────────────────────────────────

def apply_filters(audio, sr, lowcut=None, highcut=None):
    if lowcut and highcut:
        b, a = signal.butter(4, [lowcut / (sr / 2), highcut / (sr / 2)], btype='band')
        return signal.lfilter(b, a, audio)
    elif lowcut:
        b, a = signal.butter(4, lowcut / (sr / 2), btype='high')
        return signal.lfilter(b, a, audio)
    elif highcut:
        b, a = signal.butter(4, highcut / (sr / 2), btype='low')
        return signal.lfilter(b, a, audio)
    return audio


def apply_fade(audio, sr, fade_in_sec, fade_out_sec):
    n = len(audio)
    fi = int(fade_in_sec * sr)
    fo = int(fade_out_sec * sr)

    if audio.ndim == 1:
        if fi > 0:
            audio[:fi] *= np.linspace(0, 1, fi)
        if fo > 0:
            audio[-fo:] *= np.linspace(1, 0, fo)
    else:
        if fi > 0:
            audio[:fi] *= np.linspace(0, 1, fi)[:, np.newaxis]
        if fo > 0:
            audio[-fo:] *= np.linspace(1, 0, fo)[:, np.newaxis]

    return audio


def normalize_and_volume(audio, volume):
    if audio.ndim == 1:
        max_val = np.max(np.abs(audio))
    else:
        max_val = np.max(np.abs(audio))

    if max_val > 0:
        audio = audio / max_val

    return audio * volume


def to_stereo(audio):
    """Convert mono to stereo with slight variation between channels."""
    if audio.ndim == 2:
        return audio
    # Add tiny decorrelation between L and R for a wider sound
    noise_l = np.random.normal(0, 0.002, len(audio))
    noise_r = np.random.normal(0, 0.002, len(audio))
    left  = audio + noise_l
    right = audio + noise_r
    return np.stack([left, right], axis=1)


def process_audio(audio, sr, args, is_preview=False):
    """Apply all post-processing: filters, fade, volume, stereo."""

    # Apply custom filters if set
    if args.lowcut or args.highcut:
        if audio.ndim == 1:
            audio = apply_filters(audio, sr, args.lowcut, args.highcut)
        else:
            audio[:, 0] = apply_filters(audio[:, 0], sr, args.lowcut, args.highcut)
            audio[:, 1] = apply_filters(audio[:, 1], sr, args.lowcut, args.highcut)

    # Normalize and apply volume
    audio = normalize_and_volume(audio, args.volume)

    # Stereo conversion
    if not args.mono and args.noise_type != "binaural":
        audio = to_stereo(audio)

    # Fade in/out
    fade_in  = args.fade_in  if not is_preview else min(args.fade_in, 2)
    fade_out = args.fade_out if not is_preview else min(args.fade_out, 2)
    audio = apply_fade(audio, sr, fade_in, fade_out)

    return audio.astype(np.float32)


# ─────────────────────────────────────────────
# Save wav
# ─────────────────────────────────────────────

def save_wav(audio, sr, path):
    sf.write(str(path), audio, sr, subtype='PCM_16')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    args.noise_type = args.type  # alias for use in process_audio
    total_start = time.time()

    sr = args.samplerate
    base_dir     = Path(__file__).parent.parent / "generated" / "whitenoise"
    output_dir   = base_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  White Noise Generator")
    print(f"  Type       : {args.type}")
    print(f"  Output     : {args.output}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Volume     : {args.volume}")
    print(f"  Stereo     : {not args.mono}")
    if args.type == "binaural":
        print(f"  Binaural   : {args.freq_left}Hz (L) / {args.freq_right}Hz (R)")
        print(f"  Beat freq  : {abs(args.freq_right - args.freq_left)}Hz")
    print(f"{'='*55}\n")

    # ── Step 1: Generate preview variants ──
    chosen_variant = None

    if not args.skip_preview:
        print(f"Generating {args.variants} preview variant(s) × {args.preview}s each...\n")
        preview_n = args.preview * sr
        preview_paths = []

        for i in range(1, args.variants + 1):
            t = time.time()
            print(f"  Variant {i}/{args.variants}...", end=" ", flush=True)

            raw   = generate_noise(args.type, preview_n, sr, args)
            audio = process_audio(raw, sr, args, is_preview=True)
            path  = output_dir / f"preview_{i:02d}.wav"
            save_wav(audio, sr, path)
            preview_paths.append(path)
            print(f"done in {elapsed(t)}  →  {path.name}")

        print(f"\n✓ {args.variants} preview(s) saved to: {output_dir}\n")

        # ── Step 2: Pick a variant ──
        if args.pick is not None:
            chosen_variant = args.pick
            print(f"Auto-selected variant {chosen_variant}")
        else:
            print("Listen to the preview files and choose your favourite.")
            print(f"Files are in: {output_dir}\n")
            while True:
                try:
                    choice = input(f"Enter variant number (1–{args.variants}): ").strip()
                    chosen_variant = int(choice)
                    if 1 <= chosen_variant <= args.variants:
                        break
                    else:
                        print(f"  Please enter a number between 1 and {args.variants}")
                except ValueError:
                    print("  Please enter a valid number")

        print(f"\n✓ Variant {chosen_variant} selected\n")

    # ── Step 3: Stream raw PCM to disk, then wrap with ffmpeg ──
    # Bypasses WAV 4GB limit by writing headerless raw PCM first,
    # then using ffmpeg to produce the final WAV with correct headers.
    print(f"Generating full {args.duration}s ({args.duration//3600}h {(args.duration%3600)//60}m)...")
    print(f"Strategy: stream raw PCM chunks to disk, ffmpeg wraps final file\n")

    import subprocess

    t          = time.time()
    chunk_sec  = 300
    chunk_n    = chunk_sec * sr
    total_n    = args.duration * sr
    n_chunks   = -(-total_n // chunk_n)
    n_channels = 1 if args.mono else 2

    raw_path   = output_dir / f"{args.output}_raw.pcm"
    final_path = output_dir / f"{args.output}_{args.type}_{args.duration//3600}h.wav"

    temp_args            = argparse.Namespace(**vars(args))
    temp_args.noise_type = args.type
    temp_args.fade_in    = 0
    temp_args.fade_out   = 0

    samples_written = 0

    # Write raw PCM directly — no header, no size limit
    with open(raw_path, 'wb') as pcm_file:
        for c in range(n_chunks):
            remaining = total_n - samples_written
            this_n    = min(chunk_n, remaining)
            is_first  = (c == 0)
            is_last   = (c == n_chunks - 1)

            print(f"  Chunk {c+1}/{n_chunks}  ({this_n // sr}s)...", end=" ", flush=True)
            tc = time.time()

            raw   = generate_noise(args.type, this_n, sr, temp_args)
            audio = process_audio(raw, sr, temp_args, is_preview=False)

            # Fade in on first chunk only
            if is_first and args.fade_in > 0:
                fi = min(int(args.fade_in * sr), len(audio))
                if audio.ndim == 1:
                    audio[:fi] *= np.linspace(0, 1, fi)
                else:
                    audio[:fi] *= np.linspace(0, 1, fi)[:, np.newaxis]

            # Fade out on last chunk only
            if is_last and args.fade_out > 0:
                fo = min(int(args.fade_out * sr), len(audio))
                if audio.ndim == 1:
                    audio[-fo:] *= np.linspace(1, 0, fo)
                else:
                    audio[-fo:] *= np.linspace(1, 0, fo)[:, np.newaxis]

            # Convert float32 → int16 and write raw bytes
            pcm = np.clip(audio, -1.0, 1.0)
            pcm = (pcm * 32767).astype(np.int16)
            pcm_file.write(pcm.tobytes())

            samples_written += this_n
            print(f"done in {elapsed(tc)}")

    print(f"  ✓ All {n_chunks} chunks written — wrapping with ffmpeg...")

    # Use ffmpeg to convert raw PCM → proper WAV (handles >4GB via RF64)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(sr),
        "-ac", str(n_channels),
        "-i", str(raw_path),
        "-c:a", "pcm_s16le",
        "-rf64", "auto",   # automatically uses RF64/W64 format for >4GB
        str(final_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ ffmpeg wrap failed:\n{result.stderr}")
        sys.exit(1)

    # Clean up raw PCM
    raw_path.unlink()
    print(f"  ✓ Wrapped successfully")

    file_size = final_path.stat().st_size / (1024 ** 2) if final_path.stat().st_size < 1024**3 else final_path.stat().st_size / (1024 ** 3)
    unit      = "MB" if final_path.stat().st_size < 1024**3 else "GB"
    print(f"\n✓ Full file generated in {elapsed(t)}")
    print(f"  Duration  : {args.duration//3600}h {(args.duration%3600)//60}m")
    print(f"  File size : {file_size:.2f} {unit}")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Output     : {final_path}")
    print(f"\n  Next step — assemble video:")
    print(f"  python3 scripts/assemble.py --video_name whitenoise/{args.output} --animation assets/your_loop.mp4 --audio_file {final_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()