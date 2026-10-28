"""Push-to-talk and text chat conversation loop for Riko."""

import hashlib
import queue
import random
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from process.asr_func.asr_push_to_talk import (
    detect_text_language,
    read_key,
    record_and_transcribe,
    watch_for_enter,
)
from process.avatar_func.avatar_ws import (
    reset_face,
    set_emotion,
    set_state,
    set_viseme,
    start_avatar_server,
)
from process.lipsync_func.rhubarb_lipsync import generate_lipsync
from process.llm_funcs import llm_scr
from process.llm_funcs.llm_scr import EMOTIONS, char_config, save_interrupted_turn, stream_reply
from process.tts_func import piper_tts, sovits_ping
from process.tts_func.sovits_ping import Streamed, play_audio, play_stream, stop_audio
from process.tts_func.tts import can_stream, synthesize, synthesize_stream


# A sentence ends at . ! ? or … (plus closing quotes/brackets) followed by whitespace, or at a newline.
SENTENCE_END = re.compile(r"[.!?…]+[\"'»)\]]*\s|\n")
MIN_SENTENCE_CHARS = 12  # shorter pieces ("Oh!") are joined to the next sentence
MAX_SENTENCE_CHARS = 150  # longer runs are cut at a comma so audio starts sooner
FIRST_PIECE_WORDS = 4  # the first sentence may stop at a comma after this many words
EMOTION_TAG = re.compile(r"\[\s*([A-Za-z]+)\s*\]")
BRACKETED = re.compile(r"\[[^\[\]]{0,60}\]")  # any tag, even a malformed one, is never spoken
# Moods Claude sometimes invents, mapped to the expressions the avatar has.
EMOTION_ALIASES = {
    "smug": "happy", "teasing": "happy", "excited": "happy", "proud": "happy",
    "flustered": "surprised", "embarrassed": "surprised", "shocked": "surprised",
    "annoyed": "angry", "pouting": "angry", "jealous": "angry",
    "tired": "sad", "worried": "sad", "lonely": "sad",
    "calm": "relaxed", "content": "relaxed",
}
FILLER_DELAY = 0.4  # seconds of silence after the user's turn before Riko says "Hmm..."
FILLER_DIR = Path("audio") / "fillers"
_DONE = object()
_fillers = {}  # language -> [(wav_path, visemes)]


def _remove_wav(path):
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        print(f"[AUDIO CLEANUP] Could not delete {path}: {exc}")


def _find_cut(buffer, first=False):
    # An emotion tag always starts a new sentence, so the face changes on the right words.
    tag_cut = next(
        (match.start() for match in EMOTION_TAG.finditer(buffer) if buffer[:match.start()].strip()),
        None,
    )
    if first:
        # A short first piece is synthesized quickly, so Riko starts talking sooner.
        for match in re.finditer(r",\s", buffer):
            if tag_cut is not None and match.end() > tag_cut:
                break
            if len(EMOTION_TAG.sub(" ", buffer[:match.start()]).split()) >= FIRST_PIECE_WORDS:
                sentence_end = next(
                    (end.end() for end in SENTENCE_END.finditer(buffer, 0, match.start())
                     if len(buffer[:end.end()].strip()) >= MIN_SENTENCE_CHARS),
                    None,
                )
                return sentence_end or match.end()
    for match in SENTENCE_END.finditer(buffer):
        if tag_cut is not None and match.end() > tag_cut:
            break
        if len(buffer[:match.end()].strip()) >= MIN_SENTENCE_CHARS:
            return match.end()
    if tag_cut is not None:
        return tag_cut
    if len(buffer) > MAX_SENTENCE_CHARS:
        cut = max(buffer.rfind(mark, 0, MAX_SENTENCE_CHARS) for mark in (", ", "; ", ": "))
        if cut > 0:
            return cut + 2
    return None


def split_sentences(chunks):
    """Turn a stream of text chunks into speakable sentences as soon as each one is complete."""
    buffer = ""
    first = True
    for chunk in chunks:
        buffer += chunk
        while (cut := _find_cut(buffer, first)) is not None:
            sentence, buffer = buffer[:cut].strip(), buffer[cut:]
            if EMOTION_TAG.sub("", sentence).strip():
                first = False
            if sentence:
                yield sentence
    if buffer.strip():
        yield buffer.strip()


def extract_emotion(sentence):
    """Remove [emotion] tags from a sentence; return (text to speak, last known emotion or None)."""
    emotion = None
    for match in EMOTION_TAG.finditer(sentence):
        name = match.group(1).lower()
        name = EMOTION_ALIASES.get(name, name)
        if name in EMOTIONS:
            emotion = name
    text = re.sub(r"\s{2,}", " ", BRACKETED.sub(" ", sentence)).strip()
    return text, emotion


def _validate_wav(gen_aud_path):
    if gen_aud_path is None:
        raise RuntimeError("TTS did not return a WAV path")
    wav_path = Path(gen_aud_path)
    if not wav_path.exists() or not wav_path.is_file():
        raise FileNotFoundError(f"Generated WAV is missing: {wav_path}")
    if wav_path.stat().st_size == 0:
        raise ValueError(f"Generated WAV is empty: {wav_path}")
    with sf.SoundFile(wav_path) as wav:
        if wav.format != "WAV" or wav.frames == 0 or wav.samplerate <= 0:
            raise ValueError(f"Generated audio is not a playable WAV: {wav_path}")
    return wav_path


def _make_filler(text, lang, path, tries=3, good_enough=1.2):
    """Keep the shortest of a few takes: GPT-SoVITS sometimes adds sounds to very short text."""
    best = None
    for attempt in range(tries):
        take = path.with_name(f"{path.stem}.take{attempt}.wav")
        try:
            data, samplerate = sf.read(str(_validate_wav(synthesize(text, lang, take))))
        finally:
            _remove_wav(take)
        # Trim the silence around the sound so the real answer is not held back.
        loud = np.flatnonzero(np.abs(data) > np.abs(data).max() * 0.05)
        pad = int(samplerate * 0.05)
        if len(loud):
            data = data[max(0, loud[0] - pad):loud[-1] + pad]
        if best is None or len(data) < len(best[0]):
            best = (data, samplerate)
        if len(data) / samplerate <= good_enough:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), *best)
    return path


def load_fillers():
    """Synthesize the short "Hmm..." clips once; they are cached in audio/fillers."""
    voice = char_config.get("sovits_ping_config", {})
    for lang, texts in (char_config.get("fillers") or {}).items():
        clips = []
        for text in texts:
            key = f"{lang}|{text}|{voice.get('ref_audio_path')}|{voice.get('prompt_text')}"
            path = FILLER_DIR / f"{lang}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.wav"
            try:
                if not path.is_file():
                    _make_filler(text, lang, path)
                clips.append((path, generate_lipsync(path, lang, text, mode="energy")))
            except Exception as exc:
                print(f"[FILLER WARNING] Could not prepare {text!r}: {exc}")
        if clips:
            _fillers[lang] = clips
    print(f"[FILLER] Ready: {', '.join(f'{lang} x{len(clips)}' for lang, clips in _fillers.items()) or 'none'}")


def _timed(chunks, started, stop):
    """Pass the chunks through, logging the first one, until the user cuts Riko off."""
    for index, chunk in enumerate(chunks):
        if index == 0:
            print(f"[TIMING] First token after {time.monotonic() - started:.2f} s")
        yield chunk
        if stop.is_set():
            return


def _produce_audio(user_text, lang, synthesized, started, stop, turn):
    """Stream Claude's reply and synthesize it sentence by sentence into the synthesized queue."""
    tts_failed = False
    emotion = None
    stream_next = can_stream(lang)  # only the first sentence: later ones are ready in time
    reply = None
    try:
        reply = stream_reply(user_text, turn)
        for tagged_sentence in split_sentences(_timed(reply, started, stop)):
            if stop.is_set():
                break
            sentence, new_emotion = extract_emotion(tagged_sentence)
            emotion = new_emotion or emotion
            if not sentence:
                continue  # a lone tag applies to the next sentence
            print(f"Riko{f' [{emotion}]' if emotion else ''}: {sentence}")
            if tts_failed:
                continue  # keep reading so the full reply is still saved to history

            tts_started = time.monotonic()
            if stream_next:
                stream_next = False
                streaming = []

                def start_playback(streamed, sentence=sentence, emotion=emotion):
                    print(f"[TIMING] Streamed audio started after {time.monotonic() - tts_started:.2f} s")
                    streaming.append(streamed)
                    synthesized.put((streamed, sentence, emotion))

                try:
                    synthesize_stream(sentence, start_playback)
                    print(f"[TIMING] TTS took {time.monotonic() - tts_started:.2f} s (streamed)")
                    emotion = None
                    continue
                except Exception as exc:
                    print(f"[TTS ERROR] Streaming failed: {exc}")
                    if streaming:
                        emotion = None
                        continue  # part of the sentence was already spoken
                    # nothing played yet: synthesize the sentence the usual way

            output_wav_path = Path("audio") / f"output_{uuid.uuid4().hex}.wav"
            try:
                wav_path = _validate_wav(synthesize(sentence, lang, output_wav_path))
            except Exception as exc:
                print(f"[TTS ERROR] {exc}")
                _remove_wav(output_wav_path)
                tts_failed = True
                continue
            print(f"[TIMING] TTS took {time.monotonic() - tts_started:.2f} s")

            synthesized.put((wav_path, sentence, emotion))
            emotion = None  # only send an emotion again when it changes
    except Exception as exc:
        print(f"[LLM ERROR] {exc}")
    finally:
        try:
            # Stops reading Claude's stream if the user cut Riko off.
            getattr(reply, "close", lambda: None)()
        except Exception as exc:
            print(f"[LLM ERROR] {exc}")
        synthesized.put(_DONE)


def _add_lipsync(lang, synthesized, ready):
    """Compute mouth cues in their own thread, so the next sentence is synthesized meanwhile."""
    try:
        while (item := synthesized.get()) is not _DONE:
            wav_path, sentence, emotion = item
            if isinstance(wav_path, Streamed):
                # Its mouth follows the chunks as they play.
                ready.put((wav_path, None, emotion, sentence))
                continue
            lipsync_started = time.monotonic()
            try:
                visemes = generate_lipsync(wav_path, lang, sentence)
            except Exception as exc:
                print(f"[LIPSYNC ERROR] {exc}")
                visemes = []
            print(f"[TIMING] Lip sync took {time.monotonic() - lipsync_started:.2f} s")
            ready.put((wav_path, visemes, emotion, sentence))
    finally:
        ready.put(_DONE)


def _discard(ready):
    """Delete the sentences that were still waiting after an interruption."""
    while (item := ready.get()) is not _DONE:
        if not isinstance(item[0], Streamed):
            _remove_wav(item[0])


def _first_item(ready, lang, stop, use_filler):
    """Wait for the first sentence; say a short "Hmm..." if it takes a moment."""
    clips = _fillers.get(lang) if use_filler else None
    if not clips:
        return ready.get()
    try:
        return ready.get(timeout=FILLER_DELAY)
    except queue.Empty:
        pass
    wav_path, visemes = random.choice(clips)
    print("[FILLER] Riko reacts while she thinks")
    try:
        play_audio(wav_path, visemes=visemes, viseme_callback=set_viseme, stop=stop)
    except Exception as exc:
        print(f"[AUDIO ERROR] {exc}")
    return None if stop.is_set() else ready.get()


def respond(user_text, lang="en", interruptible=False, use_filler=False):
    """Play each sentence as soon as it is ready while the next one is being synthesized.

    With interruptible, pressing ENTER cuts Riko off; returns True when that happened.
    """
    set_state("thinking")
    started = time.monotonic()
    synthesized, ready = queue.Queue(), queue.Queue()
    stop, finished = threading.Event(), threading.Event()
    turn = {}
    threading.Thread(
        target=_produce_audio,
        args=(user_text, lang, synthesized, started, stop, turn),
        daemon=True,
        name="RikoVoice",
    ).start()
    threading.Thread(
        target=_add_lipsync, args=(lang, synthesized, ready), daemon=True, name="RikoLipsync"
    ).start()

    def listen():
        if watch_for_enter(finished):
            print("✋ Interrupted")
            stop.set()
            stop_audio()

    listener = None
    if interruptible:
        listener = threading.Thread(target=listen, daemon=True, name="RikoInterrupt")
        listener.start()

    talking = False
    played_until = None
    spoken = []
    item = _first_item(ready, lang, stop, use_filler)
    try:
        while item is not None and item is not _DONE and not stop.is_set():
            wav_path, visemes, emotion, sentence = item
            try:
                if emotion:
                    set_emotion(emotion)
                if not talking:
                    print(f"[TIMING] First audio after {time.monotonic() - started:.2f} s")
                    set_state("talking")
                    talking = True
                elif played_until is not None:
                    print(f"[TIMING] Gap between sentences {time.monotonic() - played_until:.2f} s")
                spoken.append(sentence)
                if isinstance(wav_path, Streamed):
                    play_stream(wav_path, viseme_callback=set_viseme, stop=stop)
                else:
                    play_audio(wav_path, visemes=visemes, viseme_callback=set_viseme, stop=stop)
                played_until = time.monotonic()
            except Exception as exc:
                print(f"[AUDIO ERROR] {exc}")
            finally:
                if not isinstance(wav_path, Streamed):
                    _remove_wav(wav_path)
            item = None if stop.is_set() else ready.get()
    finally:
        finished.set()
        if listener:
            listener.join()  # gives the terminal back before the next recording
        if stop.is_set():
            if item is not _DONE:
                if item is not None and not isinstance(item[0], Streamed):
                    _remove_wav(item[0])
                # The rest is still being synthesized; clean it up without waiting.
                threading.Thread(target=_discard, args=(ready,), daemon=True).start()
            try:
                save_interrupted_turn(user_text, " ".join(spoken), turn)
            except Exception as exc:
                print(f"[LLM ERROR] Could not save the interrupted reply: {exc}")
        set_viseme("RESET")
        set_state("idle")
        reset_face()
    return stop.is_set()


def wait_for_mode():
    """Return "voice" on ENTER or "chat" on T."""
    print("Press ENTER to start recording, or T to type...")
    while True:
        key = read_key()
        if key in ("\r", "\n"):
            return "voice"
        if key.lower() == "t":
            return "chat"


def chat_mode():
    """Send typed messages to Riko until the user returns to voice mode."""
    set_state("idle")
    print("💬 Chat mode. Type /v or press Ctrl-D to go back to voice mode.")
    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            break
        if user_text.lower() == "/v":
            break
        if user_text:
            respond(user_text, detect_text_language(user_text))
    print("🎙️  Voice mode.")


def main():
    print("\n========= Starting Riko... =========\n")
    llm_scr.warm_up()
    try:
        piper_tts.warm_up()
    except Exception as exc:
        print(f"[TTS WARNING] French voice unavailable: {exc}")
    try:
        sovits_ping.warm_up()
    except Exception as exc:
        print(f"[TTS WARNING] GPT-SoVITS warm-up failed: {exc}")
    load_fillers()
    start_avatar_server()
    asr_config = char_config.get("asr", {})

    def load_whisper(name):
        return WhisperModel(
            name,
            device=asr_config.get("device", "cpu"),
            compute_type=asr_config.get("compute_type", "int8"),
            cpu_threads=asr_config.get("cpu_threads", 0),
        )

    whisper_model = load_whisper(asr_config.get("whisper_model", "small"))
    language_name = asr_config.get("language_model")
    language_model = load_whisper(language_name) if language_name else None

    record_now = False  # after an interruption the user is already talking
    while True:
        set_state("idle")
        if not record_now and wait_for_mode() == "chat":
            chat_mode()
            continue
        record_now = False

        set_state("listening")
        conversation_recording = Path("audio") / "conversation.wav"
        conversation_recording.parent.mkdir(parents=True, exist_ok=True)

        try:
            user_spoken_text, lang = record_and_transcribe(
                whisper_model,
                conversation_recording,
                wait_for_start=False,
                language_model=language_model,
            )
            if user_spoken_text:
                record_now = respond(
                    user_spoken_text,
                    lang,
                    interruptible=True,
                    use_filler=char_config.get("fillers_enabled", True),
                )
            else:
                set_state("idle")
        finally:
            _remove_wav(conversation_recording)


if __name__ == "__main__":
    main()
