#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

[[ -x .venv/bin/python ]] || uv venv --python 3.10

case "$(uname -s)" in
    Darwin)
        # Riko itself needs no PyTorch: Whisper runs on CTranslate2 and Piper on ONNX.
        # GPT-SoVITS keeps its own environment (see start_chat.sh), so its
        # requirements.txt, which does not build on macOS, is not installed here.
        uv pip install --python .venv/bin/python -r extra-req.txt

        for tool in ffmpeg rhubarb; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                echo "WARNING: $tool is not on PATH."
            fi
        done
        if ! command -v ffmpeg >/dev/null 2>&1; then
            echo "  Install ffmpeg with: brew install ffmpeg"
        fi
        if ! command -v rhubarb >/dev/null 2>&1; then
            echo "  Lip sync needs Rhubarb: https://github.com/DanielSWolf/rhubarb-lip-sync/releases"
        fi
        ;;
    *)
        # Check your NVIDIA/CUDA version before using this CUDA 12.6 PyTorch index.
        uv pip install --python .venv/bin/python torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu126
        uv pip install --python .venv/bin/python -r extra-req.txt -r requirements.txt

        .venv/bin/python - <<'PYCODE'
import nltk
for pkg in ["averaged_perceptron_tagger", "cmudict"]:
    nltk.download(pkg)
PYCODE
        ;;
esac

# French voice for Piper (GPT-SoVITS cannot speak French).
if [[ ! -f character_files/piper/fr_FR-siwis-medium.onnx ]]; then
    mkdir -p character_files/piper
    .venv/bin/python -m piper.download_voices --download-dir character_files/piper fr_FR-siwis-medium
fi
