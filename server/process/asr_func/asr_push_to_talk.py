import os
import sys
import time

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

def read_key():
    """Read one keypress from the terminal without waiting for ENTER."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line[0]

    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def watch_for_enter(done, poll=0.05):
    """Return True if ENTER is pressed before the done event is set; False otherwise.

    Lets the user cut Riko off while she speaks. Only works in a terminal.
    """
    if not sys.stdin.isatty():
        done.wait()
        return False

    if os.name == "nt":
        import msvcrt

        while not done.is_set():
            if msvcrt.kbhit() and msvcrt.getwch() in ("\r", "\n"):
                return True
            time.sleep(poll)
        return False

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not done.is_set():
            readable, _, _ = select.select([fd], [], [], poll)
            # os.read skips Python's stdin buffer, which select() cannot see.
            if readable and os.read(fd, 1) in (b"\r", b"\n"):
                return True
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


SUPPORTED_LANGUAGES = ("en", "fr")  # English wins a tie


def pick_language(language_probs):
    """Return the most likely supported language from Whisper's (code, probability) list."""
    probs = dict(language_probs)
    return max(SUPPORTED_LANGUAGES, key=lambda language: probs.get(language, 0.0))


def detect_text_language(text):
    """Local French/English guess for typed text; English when unsure."""
    try:
        from fast_langdetect import detect

        try:
            results = detect(text.replace("\n", " "), model="lite", k=5)
        except TypeError:
            # fast_langdetect 0.x returns only the best guess as a dict.
            results = [detect(text.replace("\n", " "))]
    except Exception as exc:
        print(f"[LANG] Detection failed, using English: {exc}")
        return "en"
    scores = {result["lang"]: result["score"] for result in results}
    return "fr" if scores.get("fr", 0.0) > scores.get("en", 0.0) else "en"


# Names Whisper should spell right instead of hearing "sans paix" for "senpai".
HOTWORDS = "Senpai, Riko"


def record_and_transcribe(
    model, output_file="recording.wav", samplerate=16000, wait_for_start=True, language_model=None
):
    """
    Push-to-talk recorder: record -> save -> detect French/English -> transcribe.
    language_model, a smaller Whisper, detects the language: Whisper runs its whole
    encoder for detection, so the big model would pay that cost twice.
    Returns (text, language).
    """

    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)

    if wait_for_start:
        print("Press ENTER to start recording...")
        input()

    print("🔴 Recording... Press ENTER to stop")

    # Whisper works at 16 kHz, so record at that rate directly.
    start_time = time.monotonic()
    recording = sd.rec(int(60 * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    input()  # Wait for stop
    sd.stop()
    frames = min(len(recording), int((time.monotonic() - start_time) * samplerate))
    audio = recording[:frames, 0]

    print("⏹️  Saving audio...")
    sf.write(output_file, audio, samplerate)

    if frames < samplerate // 4:
        print("Recording too short; nothing to transcribe.")
        return "", "en"

    print("🎯 Transcribing...")
    started = time.monotonic()
    # Pick French or English first so the clip is transcribed once, in the right language.
    _, _, language_probs = (language_model or model).detect_language(audio)
    language = pick_language(language_probs)
    # Greedy decoding without timestamps is about twice as fast and as accurate on short turns.
    segments, _ = model.transcribe(
        audio, language=language, beam_size=1, without_timestamps=True, hotwords=HOTWORDS
    )
    transcription = " ".join(segment.text for segment in segments).strip()

    print(f"Transcription ({language}): {transcription}")
    print(f"[TIMING] Transcription took {time.monotonic() - started:.2f} s")
    return transcription, language


# Example usage
if __name__ == "__main__":
    model = WhisperModel("small", device="cpu", compute_type="int8")
    result = record_and_transcribe(model)
    print(f"Got: {result!r}")
