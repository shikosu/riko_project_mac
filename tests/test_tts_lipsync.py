"""No-microphone checks for response playback and Rhubarb fallback."""

import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import wave
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "server"))
# PortAudio can be unavailable in headless test environments.
sys.modules.setdefault("sounddevice", types.SimpleNamespace(play=None, wait=None))

import main_chat
from process.lipsync_func import rhubarb_lipsync
from process.asr_func import asr_push_to_talk
from process.tts_func import piper_tts, sovits_ping, tts


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        old_cwd = Path.cwd()
        os.chdir(self.temp_dir.name)
        self.addCleanup(os.chdir, old_cwd)
        streaming = patch.object(main_chat, "can_stream", return_value=False)
        streaming.start()
        self.addCleanup(streaming.stop)

    @staticmethod
    def write_wav(_text, _lang, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, [0.0] * 800, 8000)
        return path

    def test_rhubarb_failure_still_plays_and_cleans_up(self):
        events = []
        with patch.object(main_chat, "stream_reply", return_value=iter(["Hello"])), \
             patch.object(main_chat, "synthesize", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", side_effect=RuntimeError("failed")), \
             patch.object(main_chat, "set_state", side_effect=lambda value: events.append(value)), \
             patch.object(main_chat, "set_viseme", side_effect=lambda value: events.append(value)), \
             patch.object(main_chat, "reset_face", side_effect=lambda: events.append("face_reset")), \
             patch.object(main_chat, "play_audio", side_effect=lambda path, **kwargs: events.append(("play", kwargs["visemes"]))):
            main_chat.respond("Hi")

        self.assertEqual(events, ["thinking", "talking", ("play", []), "RESET", "idle", "face_reset"])
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_success_waits_for_playback_before_idle_and_cleanup(self):
        events = []
        cues = [{"start": 0.0, "end": 0.1, "viseme": "aa"}]

        def fake_play(path, **kwargs):
            self.assertEqual(kwargs["visemes"], cues)
            self.assertTrue(Path(path).is_file())
            events.append("playback_finished")

        with patch.object(main_chat, "stream_reply", return_value=iter(["Hello"])), \
             patch.object(main_chat, "synthesize", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", return_value=cues), \
             patch.object(main_chat, "set_state", side_effect=events.append), \
             patch.object(main_chat, "set_viseme", side_effect=events.append), \
             patch.object(main_chat, "reset_face", side_effect=lambda: events.append("face_reset")), \
             patch.object(main_chat, "play_audio", side_effect=fake_play):
            main_chat.respond("Hi")

        self.assertEqual(
            events,
            ["thinking", "talking", "playback_finished", "RESET", "idle", "face_reset"],
        )
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_missing_tts_file_does_not_enter_talking(self):
        events = []
        with patch.object(main_chat, "stream_reply", return_value=iter(["Hello"])), \
             patch.object(main_chat, "synthesize", return_value="missing.wav"), \
             patch.object(main_chat, "generate_lipsync") as lipsync, \
             patch.object(main_chat, "play_audio") as playback, \
             patch.object(main_chat, "set_state", side_effect=events.append), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"):
            main_chat.respond("Hi")

        self.assertEqual(events, ["thinking", "idle"])
        lipsync.assert_not_called()
        playback.assert_not_called()

    def test_sentences_play_in_order_as_they_are_ready(self):
        played, spoken = [], []

        def fake_synthesize(text, lang, path):
            spoken.append((text, lang))
            return self.write_wav(text, lang, path)

        def fake_lipsync(path, lang, _text):
            self.assertEqual(lang, "fr")
            return []

        chunks = iter(["Salut senpai. Tu as l'air ", "fatigué aujourd'hui ! ", "Repose-toi"])
        with patch.object(main_chat, "stream_reply", return_value=chunks), \
             patch.object(main_chat, "synthesize", side_effect=fake_synthesize), \
             patch.object(main_chat, "generate_lipsync", side_effect=fake_lipsync), \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_audio", side_effect=lambda path, **_: played.append(path)):
            main_chat.respond("Salut", "fr")

        self.assertEqual(
            spoken,
            [("Salut senpai.", "fr"), ("Tu as l'air fatigué aujourd'hui !", "fr"), ("Repose-toi", "fr")],
        )
        self.assertEqual(len(played), 3)
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_llm_error_returns_to_idle_without_talking(self):
        events = []

        def failing_stream(_text, _turn):
            raise RuntimeError("network down")
            yield  # pragma: no cover

        with patch.object(main_chat, "stream_reply", side_effect=failing_stream), \
             patch.object(main_chat, "synthesize") as synth, \
             patch.object(main_chat, "set_state", side_effect=events.append), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"):
            main_chat.respond("Hi")

        self.assertEqual(events, ["thinking", "idle"])
        synth.assert_not_called()

    def test_tts_failure_still_reads_the_whole_reply(self):
        consumed = []

        def reply(_text, _turn):
            for chunk in ["First sentence here. ", "Second sentence here."]:
                consumed.append(chunk)
                yield chunk

        with patch.object(main_chat, "stream_reply", side_effect=reply), \
             patch.object(main_chat, "synthesize", side_effect=RuntimeError("down")) as synth, \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_audio") as playback:
            main_chat.respond("Hi")

        self.assertEqual(len(consumed), 2)  # history is saved only once the stream is fully read
        self.assertEqual(synth.call_count, 1)
        playback.assert_not_called()

    def test_emotion_tags_move_the_face_and_are_not_spoken(self):
        events, spoken = [], []

        def fake_synthesize(text, lang, path):
            spoken.append(text)
            return self.write_wav(text, lang, path)

        chunks = iter(["[happy] Welcome back, senpai! ", "Still here. [sad] ", "I missed you though."])
        with patch.object(main_chat, "stream_reply", return_value=chunks), \
             patch.object(main_chat, "synthesize", side_effect=fake_synthesize), \
             patch.object(main_chat, "generate_lipsync", return_value=[]), \
             patch.object(main_chat, "set_emotion", side_effect=lambda value: events.append(("emotion", value))), \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_audio", side_effect=lambda path, **_: events.append("play")):
            main_chat.respond("Hi")

        self.assertEqual(spoken, ["Welcome back, senpai!", "Still here.", "I missed you though."])
        self.assertEqual(events, [("emotion", "happy"), "play", "play", ("emotion", "sad"), "play"])


    def test_first_sentence_streams_and_later_ones_use_files(self):
        events = []

        def fake_stream(text, on_chunk):
            streamed = sovits_ping.Streamed(8000, None)
            on_chunk(streamed)
            events.append(("streamed", text))
            return streamed

        chunks = iter(["[happy] Welcome back, senpai! ", "I missed you though."])
        with patch.object(main_chat, "can_stream", return_value=True), \
             patch.object(main_chat, "stream_reply", return_value=chunks), \
             patch.object(main_chat, "synthesize_stream", side_effect=fake_stream), \
             patch.object(main_chat, "synthesize", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", return_value=[]) as lipsync, \
             patch.object(main_chat, "set_emotion", side_effect=lambda value: events.append(("emotion", value))), \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_stream", side_effect=lambda audio, **_: events.append("play_stream")), \
             patch.object(main_chat, "play_audio", side_effect=lambda path, **_: events.append("play_file")):
            main_chat.respond("Hi")

        self.assertEqual(events[0], ("streamed", "Welcome back, senpai!"))
        self.assertEqual(events[1:], [("emotion", "happy"), "play_stream", "play_file"])
        self.assertEqual(lipsync.call_count, 1)  # the streamed sentence needs no Rhubarb pass
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_streaming_failure_before_audio_falls_back_to_a_file(self):
        played = []
        with patch.object(main_chat, "can_stream", return_value=True), \
             patch.object(main_chat, "stream_reply", return_value=iter(["Hello senpai."])), \
             patch.object(main_chat, "synthesize_stream", side_effect=RuntimeError("down")), \
             patch.object(main_chat, "synthesize", side_effect=self.write_wav) as synth, \
             patch.object(main_chat, "generate_lipsync", return_value=[]), \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_audio", side_effect=lambda path, **_: played.append(path)):
            main_chat.respond("Hi")

        synth.assert_called_once()
        self.assertEqual(len(played), 1)


    def test_enter_cuts_riko_off_and_saves_what_she_said(self):
        played, saved = [], []
        playing = threading.Event()

        def fake_play(path, stop=None, **_kwargs):
            played.append(path)
            playing.set()
            self.assertTrue(stop.wait(2))  # the listener cuts this sentence off

        def fake_watch(done):
            self.assertTrue(playing.wait(2))
            return True

        chunks = iter(["First sentence here. ", "Second sentence here. ", "Third one here."])
        with patch.object(main_chat, "stream_reply", return_value=chunks), \
             patch.object(main_chat, "synthesize", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", return_value=[]), \
             patch.object(main_chat, "watch_for_enter", side_effect=fake_watch), \
             patch.object(main_chat, "stop_audio") as stop_audio, \
             patch.object(main_chat, "save_interrupted_turn", side_effect=lambda *args: saved.append(args)), \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_audio", side_effect=fake_play):
            interrupted = main_chat.respond("Hi", interruptible=True)
            time.sleep(0.2)  # leftover sentences are deleted in the background

        self.assertTrue(interrupted)
        self.assertEqual(len(played), 1)
        stop_audio.assert_called_once()
        self.assertEqual(saved[0][:2], ("Hi", "First sentence here."))
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_without_enter_the_reply_is_not_interrupted(self):
        with patch.object(main_chat, "stream_reply", return_value=iter(["Hello senpai."])), \
             patch.object(main_chat, "synthesize", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", return_value=[]), \
             patch.object(main_chat, "watch_for_enter", side_effect=lambda done: done.wait() and False), \
             patch.object(main_chat, "save_interrupted_turn") as save, \
             patch.object(main_chat, "set_state"), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"), \
             patch.object(main_chat, "play_audio"):
            self.assertFalse(main_chat.respond("Hi", interruptible=True))
        save.assert_not_called()

    def test_filler_plays_only_when_the_reply_is_slow(self):
        filler = Path("filler.wav")
        self.write_wav("Hmm", "en", filler)
        played = []

        def slow_reply(_text, _turn):
            time.sleep(0.3)
            yield "Hello senpai."

        def fast_reply(_text, _turn):
            yield "Hello senpai."

        for reply in (slow_reply, fast_reply):
            with patch.object(main_chat, "stream_reply", side_effect=reply), \
                 patch.object(main_chat, "_fillers", {"en": [(filler, [])]}), \
                 patch.object(main_chat, "FILLER_DELAY", 0.1 if reply is slow_reply else 1), \
                 patch.object(main_chat, "synthesize", side_effect=self.write_wav), \
                 patch.object(main_chat, "generate_lipsync", return_value=[]), \
                 patch.object(main_chat, "set_state"), \
                 patch.object(main_chat, "set_viseme"), \
                 patch.object(main_chat, "reset_face"), \
                 patch.object(main_chat, "play_audio", side_effect=lambda path, **_: played.append(Path(path).name)):
                main_chat.respond("Hi", use_filler=True)

        self.assertEqual(played[0], "filler.wav")
        self.assertEqual(len(played), 3)  # filler + reply, then the fast reply alone
        self.assertNotEqual(played[2], "filler.wav")
        self.assertTrue(filler.is_file())  # fillers are reused, never deleted


class SentenceSplitTests(unittest.TestCase):
    def split(self, *chunks):
        return list(main_chat.split_sentences(iter(chunks)))

    def test_splits_across_chunk_boundaries(self):
        self.assertEqual(
            self.split("Hello sen", "pai. How are", " you? Fine"),
            ["Hello senpai.", "How are you?", "Fine"],
        )

    def test_short_pieces_join_the_next_sentence(self):
        self.assertEqual(self.split("Oh! Ah. Well, that is fine. "), ["Oh! Ah. Well, that is fine."])

    def test_decimal_points_do_not_split(self):
        self.assertEqual(self.split("It costs 3.50 euros today."), ["It costs 3.50 euros today."])

    def test_first_sentence_stops_at_a_comma_after_a_few_words(self):
        self.assertEqual(
            self.split("Oh wow, you really did that, ", "senpai, didn't you? Fine, I admit it, that is cool."),
            ["Oh wow, you really did that,", "senpai, didn't you?", "Fine, I admit it, that is cool."],
        )

    def test_first_sentence_keeps_an_earlier_full_stop(self):
        self.assertEqual(
            self.split("Hello there senpai. Wow you really did that, ", "didn't you?"),
            ["Hello there senpai.", "Wow you really did that, didn't you?"],
        )

    def test_a_lone_tag_does_not_use_up_the_short_first_piece(self):
        self.assertEqual(
            self.split("[happy]", " Oh wow you really did that, senpai, didn't you."),
            ["[happy] Oh wow you really did that,", "senpai, didn't you."],
        )

    def test_long_runs_are_cut_at_a_comma(self):
        text = "word " * 25 + ", " + "more " * 20
        sentences = self.split(text)
        self.assertGreater(len(sentences), 1)
        self.assertTrue(sentences[0].endswith(","))
        self.assertEqual(" ".join(sentences).split(), text.split())


class EmotionTagTests(unittest.TestCase):
    def test_invented_moods_map_to_avatar_expressions_and_odd_tags_are_not_spoken(self):
        self.assertEqual(main_chat.extract_emotion("[smug] Ha!"), ("Ha!", "happy"))
        self.assertEqual(
            main_chat.extract_emotion("[flustered→ use allowed tags] [surprised] Hein ?!"),
            ("Hein ?!", "surprised"),
        )

    def test_known_tags_are_returned_and_all_tags_removed(self):
        self.assertEqual(main_chat.extract_emotion("[Happy] Hi senpai!"), ("Hi senpai!", "happy"))
        self.assertEqual(main_chat.extract_emotion("Hmph [blush] whatever."), ("Hmph whatever.", None))
        self.assertEqual(main_chat.extract_emotion("[sad]"), ("", "sad"))
        self.assertEqual(main_chat.extract_emotion("No tag here."), ("No tag here.", None))


class LanguageTests(unittest.TestCase):
    def test_whisper_language_is_limited_to_french_or_english(self):
        self.assertEqual(asr_push_to_talk.pick_language([("fr", 0.6), ("en", 0.3)]), "fr")
        self.assertEqual(asr_push_to_talk.pick_language([("ca", 0.5), ("en", 0.2), ("fr", 0.1)]), "en")
        self.assertEqual(asr_push_to_talk.pick_language([("de", 0.9)]), "en")

    def test_typed_text_language(self):
        fake = types.ModuleType("fast_langdetect")
        fake.detect = lambda text, **_: [{"lang": "fr" if "ça" in text else "en", "score": 0.9}]
        with patch.dict(sys.modules, {"fast_langdetect": fake}):
            self.assertEqual(asr_push_to_talk.detect_text_language("Salut, ça va ?"), "fr")
            self.assertEqual(asr_push_to_talk.detect_text_language("How are you?"), "en")

    def test_typed_text_language_supports_old_fast_langdetect(self):
        def old_detect(text, low_memory=True):
            return {"lang": "fr", "score": 0.8}

        fake = types.ModuleType("fast_langdetect")
        fake.detect = old_detect
        with patch.dict(sys.modules, {"fast_langdetect": fake}):
            self.assertEqual(asr_push_to_talk.detect_text_language("Bonjour"), "fr")

    def test_french_uses_piper_and_english_uses_gpt_sovits(self):
        with patch.object(tts, "piper_gen", return_value="fr.wav") as piper, \
             patch.object(tts, "sovits_gen", return_value="en.wav") as sovits:
            self.assertEqual(tts.synthesize("Bonjour", "fr", "a.wav"), "fr.wav")
            self.assertEqual(tts.synthesize("Hello", "en", "b.wav"), "en.wav")
        piper.assert_called_once_with("Bonjour", "a.wav")
        sovits.assert_called_once_with("Hello", "b.wav", lang="en")

    def test_piper_writes_a_playable_wav(self):
        class FakeVoice:
            def synthesize_wav(self, _text, wav_file):
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(b"\x00\x00" * 2205)

        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(piper_tts, "_voice", return_value=FakeVoice()):
            output = piper_tts.piper_gen("Bonjour", Path(temp_dir) / "fr.wav")
            self.assertEqual(sf.info(str(output)).frames, 2205)


class AudioTests(unittest.TestCase):
    def test_playback_waits_and_resets_after_cues(self):
        events = []
        with patch.object(sovits_ping.sf, "read", return_value=([0.0] * 100, 1000)), \
             patch.object(sovits_ping.sd, "play", side_effect=lambda *_: events.append("start")), \
             patch.object(sovits_ping.sd, "wait", side_effect=lambda: events.append("wait")):
            sovits_ping.play_audio(
                "test.wav",
                visemes=[{"start": 0.0, "viseme": "aa"}],
                viseme_callback=events.append,
            )

        self.assertEqual(events, ["start", "aa", "wait", "RESET"])

    def test_bad_viseme_callback_does_not_skip_wait(self):
        events = []
        def bad_callback(_value):
            raise RuntimeError("Godot disconnected")

        with patch.object(sovits_ping.sf, "read", return_value=([0.0] * 100, 1000)), \
             patch.object(sovits_ping.sd, "play", side_effect=lambda *_: events.append("start")), \
             patch.object(sovits_ping.sd, "wait", side_effect=lambda: events.append("wait")):
            sovits_ping.play_audio(
                "test.wav",
                visemes=[{"start": 0.0, "viseme": "aa"}],
                viseme_callback=bad_callback,
            )

        self.assertEqual(events, ["start", "wait"])

    def test_sovits_stream_parses_the_header_and_pcm_chunks(self):
        header = io.BytesIO()
        with wave.open(header, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(32000)
        pcm = np.array([0, 16384, -16384], dtype="<i2").tobytes()
        # A sample split across two network reads must still decode.
        pieces = [header.getvalue()[:20], header.getvalue()[20:] + pcm[:3], pcm[3:]]
        response = types.SimpleNamespace(ok=True, iter_content=lambda chunk_size: iter(pieces), close=lambda: None)
        started = []
        with patch.object(sovits_ping._session, "post", return_value=response) as post:
            streamed = sovits_ping.sovits_stream("Hi", on_chunk=started.append)

        self.assertEqual(started, [streamed])
        self.assertEqual(streamed.samplerate, 32000)
        samples = []
        while (chunk := streamed.chunks.get_nowait()) is not None:
            samples.extend(chunk.tolist())
        self.assertEqual(samples, [0.0, 0.5, -0.5])
        self.assertEqual(post.call_args.kwargs["json"]["media_type"], "wav")
        self.assertTrue(post.call_args.kwargs["stream"])

    def test_tts_rejects_http_200_without_wav(self):
        response = types.SimpleNamespace(content=b'{"error":"bad"}', raise_for_status=lambda: None, ok=True)
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(sovits_ping._session, "post", return_value=response):
            output = Path(temp_dir) / "output.wav"
            with self.assertRaises(sf.LibsndfileError):
                sovits_ping.sovits_gen("Hello", output)
            self.assertFalse(output.exists())

    def test_tts_reports_gpt_sovits_error_body(self):
        response = types.SimpleNamespace(ok=False, status_code=400, text='{"message": "tts failed"}')
        with patch.object(sovits_ping._session, "post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "400.*tts failed"):
                sovits_ping.sovits_gen("Hello", "output.wav")

    def test_tts_rejects_reference_audio_outside_3_to_10_seconds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            long_reference = Path(temp_dir) / "long.wav"
            sf.write(long_reference, [0.0] * 8000 * 12, 8000)
            config = {**sovits_ping.char_config["sovits_ping_config"], "ref_audio_path": str(long_reference)}
            with patch.dict(sovits_ping.char_config, {"sovits_ping_config": config}), \
                 patch.object(sovits_ping._session, "post") as post:
                with self.assertRaisesRegex(ValueError, "12.0 s; GPT-SoVITS needs a 3-10 s clip"):
                    sovits_ping.sovits_gen("Hello", Path(temp_dir) / "output.wav")
            post.assert_not_called()

    def test_tts_resolves_reference_audio_from_project_root(self):
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, [0.0] * 800, 8000, format="WAV")
        response = types.SimpleNamespace(content=wav_buffer.getvalue(), raise_for_status=lambda: None, ok=True)
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(sovits_ping._session, "post", return_value=response) as post:
            output = sovits_ping.sovits_gen("Hello", Path(temp_dir) / "output.wav")
            self.assertTrue(output.is_file())

        self.assertEqual(
            post.call_args.kwargs["json"]["ref_audio_path"],
            str(PROJECT_ROOT / sovits_ping.char_config["sovits_ping_config"]["ref_audio_path"]),
        )
        self.assertEqual(post.call_args.kwargs["json"]["text_lang"], "en")
        self.assertEqual(post.call_args.kwargs["json"]["text_split_method"], "cut0")

    def test_rhubarb_mapping_and_absolute_wav_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, [0.0] * 800, 8000)

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("-o") + 1])
                self.assertEqual(command[-1], str(wav_path.resolve()))
                output.write_text(json.dumps({"mouthCues": [
                    {"start": 0.0, "end": 0.1, "value": "D"},
                    {"start": 0.1, "end": 0.2, "value": "F"},
                    {"start": 0.2, "end": 0.3, "value": "X"},
                ]}))
                return types.SimpleNamespace(returncode=0, stderr="")

            with patch.object(rhubarb_lipsync.shutil, "which", return_value="/usr/bin/rhubarb"), \
                 patch.object(rhubarb_lipsync.subprocess, "run", side_effect=fake_run):
                cues = rhubarb_lipsync.generate_lipsync(wav_path, mode="rhubarb")

        self.assertEqual([cue["viseme"] for cue in cues], ["aa", "ou", "RESET"])

    def test_rhubarb_uses_phonetic_recognizer_and_the_sentence_text(self):
        commands, dialogs = [], []

        def fake_run(command, **_kwargs):
            commands.append(command)
            if "--dialogFile" in command:
                dialogs.append(Path(command[command.index("--dialogFile") + 1]).read_text())
            Path(command[command.index("-o") + 1]).write_text('{"mouthCues": []}')
            return types.SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, [0.0] * 800, 8000)
            with patch.object(rhubarb_lipsync.shutil, "which", return_value="/usr/bin/rhubarb"), \
                 patch.object(rhubarb_lipsync.subprocess, "run", side_effect=fake_run):
                rhubarb_lipsync.generate_lipsync(wav_path, "fr", "Salut senpai.", mode="rhubarb")
                rhubarb_lipsync.generate_lipsync(wav_path, "en", mode="rhubarb")

        self.assertTrue(all("phonetic" in command for command in commands))
        self.assertEqual(dialogs, ["Salut senpai."])
        self.assertNotIn("--dialogFile", commands[1])

    def test_energy_lipsync_opens_the_mouth_only_while_she_speaks(self):
        samplerate = 16000
        t = np.arange(samplerate) / samplerate
        voice = 0.5 * np.sin(2 * np.pi * 220 * t)
        audio = np.concatenate([np.zeros(samplerate // 2), voice, np.zeros(samplerate // 2)])
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, audio, samplerate)
            with patch.object(rhubarb_lipsync.subprocess, "run") as run:
                cues = rhubarb_lipsync.generate_lipsync(wav_path, mode="energy")
            run.assert_not_called()

        self.assertEqual(cues[0], {"start": 0.0, "end": 0.48, "viseme": "RESET"})
        self.assertEqual(cues[-1]["viseme"], "RESET")
        self.assertTrue(any(cue["viseme"] != "RESET" for cue in cues))
        self.assertTrue(all(cue["end"] > cue["start"] for cue in cues))
        for previous, cue in zip(cues, cues[1:]):
            self.assertEqual(previous["end"], cue["start"])
            self.assertNotEqual(previous["viseme"], cue["viseme"])

    def test_energy_lipsync_of_silence_has_no_cues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, [0.0] * 8000, 8000)
            self.assertEqual(rhubarb_lipsync.generate_lipsync(wav_path, mode="energy"), [])

    def test_sovits_warm_up_synthesizes_once_and_keeps_no_file(self):
        paths = []

        def fake_gen(text, path):
            paths.append(Path(path))
            Path(path).write_bytes(b"x")

        with patch.object(sovits_ping, "sovits_gen", side_effect=fake_gen):
            sovits_ping.warm_up()

        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists())

    def test_enter_watcher_never_reads_a_non_terminal_stdin(self):
        done = threading.Event()
        done.set()
        with patch.object(asr_push_to_talk.sys.stdin, "isatty", return_value=False):
            self.assertFalse(asr_push_to_talk.watch_for_enter(done))

    def test_avatar_viseme_message(self):
        from process.avatar_func import avatar_ws

        with patch.object(avatar_ws, "send_avatar") as send:
            avatar_ws.set_viseme("oh")
        send.assert_called_once_with({"type": "viseme", "value": "oh"})


if __name__ == "__main__":
    unittest.main()
