"""Local French speech with Piper, since GPT-SoVITS cannot speak French."""

import io
import wave
from functools import lru_cache
from pathlib import Path

from .sovits_ping import CONFIG_FILE, char_config, save_wav_bytes


def _model_path():
    model_path = Path(char_config["piper_config"]["model_path"])
    return model_path if model_path.is_absolute() else CONFIG_FILE.parent / model_path


@lru_cache(maxsize=1)
def _voice():
    from piper import PiperVoice

    model_path = _model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Piper voice not found: {model_path} (see the README to download fr_FR-siwis-medium)"
        )
    return PiperVoice.load(str(model_path))


def warm_up():
    """Load the voice at startup instead of during the first French reply."""
    _voice()


def piper_gen(in_text, output_wav_pth="output.wav"):
    """Synthesize in_text with Piper and return the WAV path."""
    print("[TTS] Generating audio with Piper...")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _voice().synthesize_wav(in_text, wav_file)
    return save_wav_bytes(buffer.getvalue(), output_wav_pth, "Piper")
