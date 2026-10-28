"""Time speech synthesis and lip sync on fixed sentences.

Run from the project root while GPT-SoVITS is up:
    .venv/bin/python tests/benchmark_voice.py
"""

import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from process.lipsync_func.rhubarb_lipsync import generate_lipsync
from process.tts_func import piper_tts, sovits_ping
from process.tts_func.tts import synthesize


SENTENCES = {
    "en": [
        "Hmm, senpai.",
        "Wait, senpai, you built your own CPU architecture too?",
        "Don't skip the details, I want the whole nerdy story.",
        "That's way more than nothing to impress, you absolutely planned that humblebrag.",
        "Fine, I'll admit it, that is kind of cool.",
    ],
    "fr": [
        "Hein, senpai ?",
        "Tu as l'air fatigué aujourd'hui, tu devrais aller dormir.",
        "Ne me regarde pas comme ça, je ne suis pas impressionnée.",
        "Bon, d'accord, c'est un petit peu génial, mais juste un petit peu.",
        "Raconte-moi tout, je veux tous les détails.",
    ],
}


def main():
    print("Warming up...")
    sovits_ping.warm_up()
    piper_tts.warm_up()
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"{'lang':4} {'chars':>5} {'audio':>6} {'tts':>6} {'energy':>7} {'rhubarb':>7}  sentence")
        totals = {}
        for lang, sentences in SENTENCES.items():
            for index, sentence in enumerate(sentences):
                path = Path(temp_dir) / f"{lang}_{index}.wav"
                started = time.monotonic()
                synthesize(sentence, lang, path)
                tts_time = time.monotonic() - started
                started = time.monotonic()
                generate_lipsync(path, lang, sentence, mode="energy")
                energy_time = time.monotonic() - started
                started = time.monotonic()
                generate_lipsync(path, lang, sentence, mode="rhubarb")
                lipsync_time = time.monotonic() - started
                audio_time = sf.info(str(path)).duration
                total = totals.setdefault(lang, [0.0, 0.0, 0.0])
                total[0] += audio_time
                total[1] += tts_time
                total[2] += lipsync_time
                print(
                    f"{lang:4} {len(sentence):5} {audio_time:5.2f}s {tts_time:5.2f}s {energy_time:6.3f}s {lipsync_time:6.2f}s"
                    f"  {sentence}"
                )
        for lang, (audio_time, tts_time, lipsync_time) in totals.items():
            print(
                f"[{lang}] {audio_time:.1f}s of audio: TTS {tts_time:.1f}s"
                f" (x{audio_time / tts_time:.1f} real time), rhubarb {lipsync_time:.1f}s"
            )


if __name__ == "__main__":
    main()
