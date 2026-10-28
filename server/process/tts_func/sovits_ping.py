"""GPT-SoVITS synthesis and blocking WAV playback."""

import io
import queue
import tempfile
import threading
import time
import wave
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import yaml


CONFIG_FILE = Path(__file__).resolve().parents[3] / "character_config.yaml"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    char_config = yaml.safe_load(config_file)

# Audio still arriving from GPT-SoVITS: chunks is a queue of float32 arrays ending with None.
Streamed = namedtuple("Streamed", "samplerate chunks")


def stop_audio():
    """Cut off play_audio from another thread; play_stream stops on its stop event."""
    sd.stop()


def play_audio(path, visemes=None, viseme_callback=None, stop=None):
    """Play the complete WAV while dispatching Rhubarb cues at their start times.

    stop, a threading.Event, ends the visemes early; call stop_audio() to cut the sound.
    """
    data, samplerate = sf.read(path)
    if samplerate <= 0 or len(data) == 0:
        raise ValueError(f"WAV has no playable frames: {path}")

    duration = len(data) / samplerate
    print("[AUDIO] Starting playback")
    sd.play(data, samplerate)
    start_time = time.monotonic()

    try:
        if visemes and viseme_callback:
            last_viseme = None
            try:
                for cue in visemes:
                    cue_start = float(cue["start"])
                    if cue_start >= duration:
                        break

                    target = start_time + cue_start
                    remaining = target - time.monotonic()
                    while remaining > 0 and not (stop and stop.is_set()):
                        time.sleep(min(remaining, 0.01))
                        remaining = target - time.monotonic()
                    if stop and stop.is_set():
                        break

                    viseme = cue["viseme"]
                    if viseme != last_viseme:
                        viseme_callback(viseme)
                        last_viseme = viseme
            except Exception as exc:
                print(f"[LIPSYNC ERROR] Could not dispatch visemes: {exc}")

        # sd.play() is asynchronous; this must finish before returning to ASR.
        sd.wait()
        print("[AUDIO] Playback finished")
    finally:
        if viseme_callback:
            try:
                viseme_callback("RESET")
            except Exception as exc:
                print(f"[LIPSYNC ERROR] Could not reset viseme: {exc}")


def play_stream(audio, viseme_callback=None, stop=None):
    """Play a Streamed reply while its chunks arrive, moving the mouth with each chunk's loudness.

    Setting stop, a threading.Event, cuts the sound and the mouth off.
    """
    from process.lipsync_func.rhubarb_lipsync import energy_cues

    cues = queue.Queue()

    def dispatch():
        while (cue := cues.get()) is not None:
            at, viseme = cue
            while (remaining := at - time.monotonic()) > 0 and not (stop and stop.is_set()):
                time.sleep(min(remaining, 0.02))
            if stop and stop.is_set():
                break
            try:
                viseme_callback(viseme)
            except Exception as exc:
                print(f"[LIPSYNC ERROR] Could not dispatch visemes: {exc}")

    dispatcher = None
    if viseme_callback:
        dispatcher = threading.Thread(target=dispatch, daemon=True, name="RikoStreamVisemes")
        dispatcher.start()

    print("[AUDIO] Starting streamed playback")
    written = 0
    started = None
    peak = 0.0
    try:
        with sd.OutputStream(samplerate=audio.samplerate, channels=1, dtype="float32") as stream:
            while (chunk := audio.chunks.get()) is not None:
                if stop and stop.is_set():
                    break
                now = time.monotonic()
                if started is None or now > started + written / audio.samplerate:
                    # First chunk, or the network fell behind playback: restart the clock.
                    started = now + stream.latency - written / audio.samplerate
                if viseme_callback and len(chunk):
                    peak = max(peak, float(np.percentile(np.abs(chunk), 99)) * 0.5)
                    offset = started + written / audio.samplerate
                    last = None
                    for cue in energy_cues(chunk, audio.samplerate, peak):
                        if cue["viseme"] != last:
                            cues.put((offset + cue["start"], cue["viseme"]))
                            last = cue["viseme"]
                stream.write(chunk.reshape(-1, 1))
                written += len(chunk)
            if stop and stop.is_set():
                stream.abort()
            elif started is not None:
                while (remaining := started + written / audio.samplerate - time.monotonic()) > 0:
                    if stop and stop.is_set():
                        stream.abort()
                        break
                    time.sleep(min(remaining, 0.02))
        print("[AUDIO] Playback finished")
    finally:
        if dispatcher:
            cues.put(None)
            dispatcher.join(timeout=1)
            try:
                viseme_callback("RESET")
            except Exception as exc:
                print(f"[LIPSYNC ERROR] Could not reset viseme: {exc}")


SOVITS_URL = "http://127.0.0.1:9880/tts"
_session = requests.Session()  # keeps the connection open between sentences


@lru_cache(maxsize=None)
def _check_reference_audio(path):
    """GPT-SoVITS rejects a missing reference or one outside 3-10 seconds with a bare 400."""
    if not path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {path} (check ref_audio_path)")
    duration = sf.info(str(path)).duration
    if not 3 <= duration <= 10:
        raise ValueError(
            f"Reference audio is {duration:.1f} s; GPT-SoVITS needs a 3-10 s clip: {path}"
        )


def save_wav_bytes(content, output_wav_pth, engine):
    """Write WAV bytes atomically after checking they hold playable audio."""
    if not content:
        raise RuntimeError(f"{engine} returned empty audio")

    output_path = Path(output_wav_pth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".wav", dir=output_path.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise ValueError(f"{engine} wrote an empty WAV")
        with sf.SoundFile(temp_path) as wav:
            if wav.format != "WAV" or wav.frames == 0 or wav.samplerate <= 0:
                raise ValueError(f"{engine} returned a non-playable WAV")

        temp_path.replace(output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"[TTS] WAV generated: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def _payload(in_text, lang):
    config = char_config["sovits_ping_config"]
    ref_audio_path = Path(config["ref_audio_path"])
    if not ref_audio_path.is_absolute():
        ref_audio_path = CONFIG_FILE.parent / ref_audio_path
    ref_audio_path = ref_audio_path.resolve()
    _check_reference_audio(ref_audio_path)
    payload = {
        **(config.get("speed_params") or {}),
        "text": in_text,
        "text_lang": lang or config["text_lang"],
        "ref_audio_path": str(ref_audio_path),
        "prompt_text": config["prompt_text"],
        "prompt_lang": config["prompt_lang"],
    }
    return payload


def _post(payload, **kwargs):
    try:
        response = _session.post(SOVITS_URL, json=payload, timeout=120, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"GPT-SoVITS request failed at {SOVITS_URL}: {exc}") from exc
    if not response.ok:
        # GPT-SoVITS explains 400 errors in the body (e.g. a bad reference clip).
        raise RuntimeError(f"GPT-SoVITS returned {response.status_code}: {response.text.strip()[:500]}")
    return response


def sovits_gen(in_text, output_wav_pth="output.wav", lang=None):
    """Request a WAV from GPT-SoVITS, validate it, and return its path."""
    print("[TTS] Generating audio with GPT-SoVITS...")
    response = _post(_payload(in_text, lang))
    return save_wav_bytes(response.content, output_wav_pth, "GPT-SoVITS")


STREAMING_MODE = char_config["sovits_ping_config"].get("streaming_mode", 0)


def sovits_stream(in_text, lang=None, on_chunk=None):
    """Start streaming in_text from GPT-SoVITS; return a Streamed once the first audio arrives.

    The rest of the reply is read in the calling thread, so this returns only when
    GPT-SoVITS is done; on_chunk(streamed) is called as soon as playback can start.
    """
    payload = {**_payload(in_text, lang), "streaming_mode": STREAMING_MODE, "media_type": "wav"}
    print("[TTS] Streaming audio from GPT-SoVITS...")
    response = _post(payload, stream=True)
    pending = b""
    streamed = None
    try:
        for data in response.iter_content(chunk_size=None):
            pending += data
            if streamed is None:
                if len(pending) < 44:
                    continue
                # The first piece is a WAV header with no frames; the rest is 16-bit PCM.
                with wave.open(io.BytesIO(pending[:44]), "rb") as header:
                    if header.getsampwidth() != 2 or header.getnchannels() != 1:
                        raise ValueError("GPT-SoVITS streamed audio is not 16-bit mono")
                    streamed = Streamed(header.getframerate(), queue.Queue())
                pending = pending[44:]
                if on_chunk:
                    on_chunk(streamed)
            usable = len(pending) - len(pending) % 2
            if usable:
                samples = np.frombuffer(pending[:usable], dtype="<i2").astype("float32") / 32768
                streamed.chunks.put(samples)
                pending = pending[usable:]
        if streamed is None:
            raise RuntimeError("GPT-SoVITS streamed no audio")
    finally:
        response.close()
        if streamed is not None:
            streamed.chunks.put(None)
    return streamed


def warm_up():
    """Synthesize once at startup: GPT-SoVITS spends ~4 s on the reference clip the first time."""
    with tempfile.TemporaryDirectory(prefix="riko_warmup_") as temp_dir:
        sovits_gen("Hi senpai.", Path(temp_dir) / "warmup.wav")


if __name__ == "__main__":
    start_time = time.monotonic()
    path_to_audio = sovits_gen(
        "if you hear this, that means it is set up correctly", "output.wav"
    )
    print(f"Elapsed time: {time.monotonic() - start_time:.4f} seconds")
    print(path_to_audio)
