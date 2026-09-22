#!/usr/bin/env bash
# =============================================================================
# OSV-5M Setup — use the 65M-image geolocation model WITHOUT downloading images
# =============================================================================
#
# KEY INSIGHT:
#   OSV-5M's 65 MILLION streetview images are its TRAINING/benchmark corpus
#   (the `osv5m/osv5m` HF dataset, TERABYTES). To USE the model for geolocation
#   you do NOT download those images — they are distilled into the pretrained
#   weights during training. You only need:
#
#     1. the small OSV-5M CODE repo (~57 MB, from gastruc/osv5m @ github)
#     2. one pretrained model file (~700 MB pytorch_model.bin, osv5m/baseline
#        on HuggingFace, auto-fetched by `Geolocalizer.from_pretrained(...)`)
#
#   The image corpus is downloaded ONLY if you later train/fine-tune yourself,
#   which this script never does.
#
# This script wires that up for GeoVision's modules/osv5m_predictor.py.
# It does NOT download any part of the 65M-image corpus.
#
# Usage:
#   bash scripts/setup_osv5m.sh            # clone code + verify model resolves
#   bash scripts/setup_osv5m.sh --no-verify # skip the ~700MB model download
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."  # OSV5M_CODE → repo root /osv5m

VERIFY=1
[[ "${1:-}" == "--no-verify" ]] && VERIFY=0

echo "== OSV-5M setup (NO image downloads — model weights only) =="

# 1. Clone the small code repo if not present
if [[ ! -d "osv5m" ]]; then
  echo ">> Cloning gastruc/osv5m code repo (~57 MB)..."
  git clone --depth 1 https://github.com/gastruc/osv5m.git osv5m
else
  echo ">> osv5m/ code repo already present."
fi

# 2. Install the runtime deps the Geolocalizer needs (hydra-core, omegaconf).
echo ">> Ensuring runtime deps (hydra-core, omegaconf)..."
python3 -m pip install --quiet hydra-core omegaconf huggingface_hub 2>/dev/null || \
  echo "   (pip install emitted warnings — check python env is active)"

# 3. Export the path the predictor expects (persist for the shell session)
export OSV5M_PATH="$(pwd)/osv5m"
echo ">> OSV5M_PATH=$OSV5M_PATH"

if [[ "$VERIFY" == "1" ]]; then
  echo ""
  echo ">> Verifying Geolocalizer.from_pretrained('osv5m/baseline')..."
  echo "   This downloads ONLY the ~700 MB model weights (never the image corpus)."
  echo "   (one-time; cached to ~/.cache/huggingface after first run)"
  cd osv5m
  OSV5M_PATH="$OSV5M_PATH" python3 -c "
import sys; sys.path.insert(0, '.')
from models.huggingface import Geolocalizer
m = Geolocalizer.from_pretrained('osv5m/baseline')
print('✓ GeoLocalizer loaded. model type:', type(m).__name__)
print('✓ OSV-5M ready. 65M images NOT required.')
" || {
    echo "!! Model download/load failed. If this is a transient HF issue, re-run."
    echo "   (Check network; the model caches after first successful load.)"
  }
else
  echo ">> Skipping model download (--no-verify)."
fi

echo ""
echo "== Done. modules/osv5m_predictor.py will auto-load OSV-5M when OSV5M_PATH"
echo "   is set and the model is cached. 65M images never downloaded. =="