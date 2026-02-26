#!/bin/bash
# =============================================================
#  setup.sh — MusicGen Pipeline Full Setup
#  Run this once on a fresh Ubuntu/WSL2 environment.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
#
#  What it does:
#    1. Updates system packages
#    2. Installs Python 3.10, ffmpeg, git
#    3. Creates the project folder structure
#    4. Creates a virtual environment
#    5. Installs PyTorch (CUDA 12.1)
#    6. Installs AudioCraft from source
#    7. Installs all pipeline dependencies
#    8. Pins numpy and transformers to compatible versions
#    9. Creates a sample prompts.txt
#   10. Verifies the installation
# =============================================================

set -e  # Exit immediately on any error

# ─────────────────────────────────────────────
# Colors for output
# ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
info() { echo -e "${CYAN}  → $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

section() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
}

# ─────────────────────────────────────────────
# Project config
# ─────────────────────────────────────────────
PROJECT_DIR="$HOME/musicgen_pipeline"
VENV_DIR="$PROJECT_DIR/venv"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
GENERATED_DIR="$PROJECT_DIR/generated"
ASSETS_DIR="$PROJECT_DIR/assets"

# ─────────────────────────────────────────────
# 1. System packages
# ─────────────────────────────────────────────
section "Step 1 — System packages"

info "Updating package list..."
sudo apt update -y

info "Installing Python 3.10, ffmpeg, git, build tools..."
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    curl

ok "System packages installed"

# ─────────────────────────────────────────────
# 2. Verify GPU
# ─────────────────────────────────────────────
section "Step 2 — GPU check"

if command -v nvidia-smi &> /dev/null; then
    ok "nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warn "nvidia-smi not found. GPU may not be available."
    warn "PyTorch will fall back to CPU automatically."
fi

# ─────────────────────────────────────────────
# 3. Create project folder structure
# ─────────────────────────────────────────────
section "Step 3 — Project folder structure"

mkdir -p "$PROJECT_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$GENERATED_DIR"
mkdir -p "$ASSETS_DIR"

ok "Folder structure created:"
echo "     $PROJECT_DIR"
echo "     $SCRIPTS_DIR"
echo "     $GENERATED_DIR"
echo "     $ASSETS_DIR"

# ─────────────────────────────────────────────
# 4. Virtual environment
# ─────────────────────────────────────────────
section "Step 4 — Virtual environment"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR"
    warn "Deleting and recreating it..."
    rm -rf "$VENV_DIR"
fi

info "Creating virtual environment..."
python3.10 -m venv "$VENV_DIR"

# Activate
source "$VENV_DIR/bin/activate"

info "Upgrading pip..."
pip install --upgrade pip

ok "Virtual environment ready at $VENV_DIR"

# ─────────────────────────────────────────────
# 5. PyTorch (CUDA 12.1)
# ─────────────────────────────────────────────
section "Step 5 — PyTorch (CUDA 12.1)"

info "Installing PyTorch 2.5.1 with CUDA 12.1 support..."
info "This is a large download (~2GB), please wait..."

pip install \
    "torch==2.5.1+cu121" \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

ok "PyTorch installed"

# ─────────────────────────────────────────────
# 6. AudioCraft from source
# ─────────────────────────────────────────────
section "Step 6 — AudioCraft (from source)"

info "Installing AudioCraft from GitHub..."
info "This bypasses strict version pins..."

pip install git+https://github.com/facebookresearch/audiocraft.git

ok "AudioCraft installed"

# ─────────────────────────────────────────────
# 7. Pipeline dependencies
# ─────────────────────────────────────────────
section "Step 7 — Pipeline dependencies"

info "Installing pydub, moviepy, librosa, tqdm, soundfile..."

pip install \
    pydub \
    moviepy \
    librosa \
    tqdm \
    soundfile

ok "Pipeline dependencies installed"

# ─────────────────────────────────────────────
# 8. Pin numpy and transformers
# ─────────────────────────────────────────────
section "Step 8 — Pinning numpy and transformers"

info "Pinning numpy < 2.0 (required for AudioCraft compatibility)..."
pip install "numpy<2" --force-reinstall

info "Pinning transformers to compatible version..."
pip install "transformers>=4.31.0,<4.46.0" --force-reinstall

info "Pinning typing-extensions..."
pip install "typing-extensions>=4.12.0" --force-reinstall

ok "Versions pinned"

# ─────────────────────────────────────────────
# 9. Sample prompts.txt
# ─────────────────────────────────────────────
section "Step 9 — Sample prompts file"

PROMPTS_FILE="$PROJECT_DIR/prompts.txt"

if [ -f "$PROMPTS_FILE" ]; then
    warn "prompts.txt already exists, skipping sample creation"
else
    cat > "$PROMPTS_FILE" << 'EOF'
# MusicGen Prompt File
# Format: prompt text | duration_seconds | output_filename
# Duration: 5 to 360 seconds
# Lines starting with # are ignored

# ── Lofi Hip Hop ──
lofi hip hop, relaxed, 85 BPM, vinyl texture, rain sounds | 30 | lofi_001
lofi hip hop, melancholic, piano, soft drums, night city vibes | 30 | lofi_002
lofi hip hop, chill, lo-fi beats, warm bass, study music | 30 | lofi_003
lofi hip hop, jazzy chords, mellow, late night, cozy | 30 | lofi_004

# ── Ambient ──
dark ambient, cinematic, slow, deep bass, tension building | 45 | ambient_001
peaceful ambient, nature sounds, soft synth pads, calm, meditative | 45 | ambient_002
space ambient, ethereal, floating, deep cosmos, slow evolving | 45 | ambient_003

# ── Jazz ──
upbeat jazz, piano, trumpet, 120 BPM, energetic, swing | 30 | jazz_001
smooth jazz, saxophone, late night bar, relaxed, 90 BPM | 30 | jazz_002
EOF

    ok "Sample prompts.txt created at $PROMPTS_FILE"
fi

# ─────────────────────────────────────────────
# 10. Verification
# ─────────────────────────────────────────────
section "Step 10 — Verification"

info "Checking Python version..."
python3 --version

info "Checking PyTorch and GPU..."
python3 - << 'PYCHECK'
import torch
import numpy
print(f"  PyTorch   : {torch.__version__}")
print(f"  NumPy     : {numpy.__version__}")
print(f"  GPU       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
PYCHECK

info "Checking ffmpeg..."
ffmpeg -version 2>&1 | head -n 1

info "Checking AudioCraft import..."
python3 -c "from audiocraft.models import MusicGen; print('  AudioCraft: OK')"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
section "Setup Complete"

echo ""
echo -e "${GREEN}  Everything is installed and ready.${NC}"
echo ""
echo "  Project location : $PROJECT_DIR"
echo "  Scripts          : $SCRIPTS_DIR"
echo "  Prompts file     : $PROJECT_DIR/prompts.txt"
echo ""
echo "  To activate the environment in future sessions:"
echo -e "  ${CYAN}cd $PROJECT_DIR && source venv/bin/activate${NC}"
echo ""
echo "  To run the full pipeline:"
echo -e "  ${CYAN}python3 scripts/run.py --model large --prompts prompts.txt --video_name my_video --animation assets/loop.mp4${NC}"
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"