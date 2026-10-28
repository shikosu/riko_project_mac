"""Pick the speech engine for a reply language."""

from .piper_tts import piper_gen
from .sovits_ping import STREAMING_MODE, sovits_gen, sovits_stream


def synthesize(text, lang, output_wav_pth):
    """French goes to Piper; everything else to Riko's GPT-SoVITS voice."""
    if lang == "fr":
        return piper_gen(text, output_wav_pth)
    return sovits_gen(text, output_wav_pth, lang="en")


def can_stream(lang):
    """Piper is fast enough already; GPT-SoVITS streams when streaming_mode is set."""
    return lang != "fr" and bool(STREAMING_MODE)


def synthesize_stream(text, on_chunk):
    """Stream Riko's GPT-SoVITS voice; on_chunk(streamed) fires when playback can start."""
    return sovits_stream(text, lang="en", on_chunk=on_chunk)
