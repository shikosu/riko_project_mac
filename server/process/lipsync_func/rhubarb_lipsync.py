"""Mouth cues for Riko's VRM visemes, from Rhubarb or from the loudness of the WAV."""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


CONFIG_FILE = Path(__file__).resolve().parents[3] / "character_config.yaml"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    LIPSYNC_MODE = (yaml.safe_load(config_file) or {}).get("lipsync", "energy")
if LIPSYNC_MODE not in {"energy", "rhubarb"}:
    raise ValueError("lipsync must be energy or rhubarb in character_config.yaml")

ENERGY_FRAME = 0.06  # seconds per mouth shape


RHUBARB_TO_VRM = {
    "A": "RESET",
    "B": "ih",
    "C": "ee",
    "D": "aa",
    "E": "oh",
    "F": "ou",
    "G": "ih",
    "H": "ih",
    "X": "RESET",
}


def generate_lipsync(audio_path, lang="en", text=None, mode=None):
    """Return mouth cues for the WAV; text, when known, helps Rhubarb."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"WAV is missing or empty: {audio_path}")
    if (mode or LIPSYNC_MODE) == "energy":
        return energy_lipsync(audio_path)
    return rhubarb_lipsync(audio_path, text)


def energy_lipsync(audio_path):
    """Open the mouth with the loudness and shape it with the brightness of each frame.

    Takes a few milliseconds, so the first sentence never waits for lip sync.
    """
    data, samplerate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    cues = energy_cues(data.mean(axis=1), samplerate)
    print(f"[LIPSYNC] {len(cues)} mouth cues from loudness")
    return cues


def energy_cues(samples, samplerate, peak=None):
    """Mouth cues for an array of samples; peak sets the loudness of a wide-open mouth."""
    frame = max(1, int(samplerate * ENERGY_FRAME))
    count = len(samples) // frame
    if count == 0:
        return []
    frames = np.asarray(samples[:count * frame], dtype="float32").reshape(count, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = peak or np.percentile(rms, 95)
    if peak <= 1e-6:
        return []
    levels = rms / peak
    # Zero crossings separate bright sounds (ee, ih) from dark ones (oh, ou).
    crossings = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    voiced = levels > 0.15
    bright = crossings > (np.median(crossings[voiced]) if np.any(voiced) else 0)

    cues = []
    for index, (level, is_bright) in enumerate(zip(levels, bright)):
        if level < 0.15:
            viseme = "RESET"
        elif level > 0.7:
            viseme = "aa"
        elif level > 0.35:
            viseme = "ee" if is_bright else "oh"
        else:
            viseme = "ih" if is_bright else "ou"
        start = round(index * ENERGY_FRAME, 3)
        end = round((index + 1) * ENERGY_FRAME, 3)
        if cues and cues[-1]["viseme"] == viseme:
            cues[-1]["end"] = end
        else:
            cues.append({"start": start, "end": end, "viseme": viseme})
    return cues


def rhubarb_lipsync(audio_path, text=None):
    rhubarb = shutil.which("rhubarb")
    if rhubarb is None:
        raise RuntimeError("Rhubarb executable is not on PATH")

    print("[LIPSYNC] Running Rhubarb...")
    with tempfile.TemporaryDirectory(prefix="riko_rhubarb_") as temp_dir:
        json_path = Path(temp_dir) / "mouth_cues.json"
        try:
            # The default PocketSphinx recognizer takes ~4 s per second of audio;
            # phonetic is ~10x faster and works for French too.
            command = [rhubarb, "-f", "json", "-o", str(json_path), "--recognizer", "phonetic"]
            if text:
                dialog_path = Path(temp_dir) / "dialog.txt"
                dialog_path.write_text(text, encoding="utf-8")
                command += ["--dialogFile", str(dialog_path)]
            result = subprocess.run(
                [*command, str(audio_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Rhubarb timed out after 120 seconds") from exc

        if result.returncode != 0:
            raise RuntimeError(f"Rhubarb exited {result.returncode}: {result.stderr.strip()}")
        if not json_path.is_file():
            raise RuntimeError("Rhubarb did not create its JSON output")

        with json_path.open("r", encoding="utf-8") as output_file:
            data = json.load(output_file)

    if not isinstance(data, dict) or not isinstance(data.get("mouthCues"), list):
        raise ValueError("Rhubarb JSON has no mouthCues list")

    cues = []
    for cue in data["mouthCues"]:
        if not isinstance(cue, dict):
            raise ValueError("Rhubarb returned a malformed mouth cue")
        start = float(cue["start"])
        end = float(cue["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Rhubarb returned invalid cue timing: {cue}")
        cues.append({
            "start": start,
            "end": end,
            "viseme": RHUBARB_TO_VRM.get(cue.get("value"), "RESET"),
        })

    cues.sort(key=lambda cue: cue["start"])
    print(f"[LIPSYNC] {len(cues)} mouth cues generated")
    return cues
