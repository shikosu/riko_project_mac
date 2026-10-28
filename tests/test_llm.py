import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from server.process.llm_funcs import llm_scr

REAL_CLIENT = llm_scr._claude_client


class FakeStream:
    def __init__(self, chunks, stop_reason, stop_details=None):
        self.text_stream = iter(chunks)
        self.final = types.SimpleNamespace(stop_reason=stop_reason, stop_details=stop_details)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get_final_message(self):
        return self.final


class ClaudeResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.history_file = Path(self.temp_dir.name) / "chat_history.json"
        self.memory_file = Path(self.temp_dir.name) / "memory.json"
        for name, value in (("HISTORY_FILE", self.history_file), ("MEMORY_FILE", self.memory_file)):
            file_patch = patch.object(llm_scr, name, value)
            file_patch.start()
            self.addCleanup(file_patch.stop)
        background_patch = patch.object(llm_scr, "remember_in_background")
        self.remember = background_patch.start()
        self.addCleanup(background_patch.stop)

        self.request = None
        self.chunks = ["Bonjour ", "senpai."]
        self.stop_reason = "end_turn"
        self.stop_details = None

        def stream(**kwargs):
            self.request = kwargs
            return FakeStream(self.chunks, self.stop_reason, self.stop_details)

        self.summary_requests = []
        self.summary_text = "The user is called Teo and loves ramen."
        self.summary_stop_reason = "end_turn"

        def create(**kwargs):
            self.summary_requests.append(kwargs)
            return types.SimpleNamespace(
                stop_reason=self.summary_stop_reason,
                content=[types.SimpleNamespace(type="text", text=self.summary_text)],
            )

        client = types.SimpleNamespace(
            beta=types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream, create=create))
        )
        client_patch = patch.object(llm_scr, "_claude_client", return_value=client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def saved(self):
        return json.loads(self.history_file.read_text(encoding="utf-8"))

    def test_streams_chunks_and_saves_reply(self):
        chunks = list(llm_scr.stream_reply("Salut"))

        self.assertEqual(chunks, ["Bonjour ", "senpai."])
        self.assertEqual(self.request["model"], llm_scr.MODEL)
        self.assertEqual(self.request["system"], llm_scr.SYSTEM_PROMPT_TEXT)
        self.assertEqual(self.request["output_config"], {"effort": llm_scr.EFFORT})
        self.assertEqual(self.request["fallbacks"], "default")
        self.assertEqual(self.request["betas"], [llm_scr.FALLBACK_BETA])
        self.assertNotIn("temperature", self.request)
        self.assertEqual(len(self.request["messages"]), 1)
        self.assertEqual(self.request["messages"][0]["role"], "user")
        self.assertRegex(self.request["messages"][0]["content"], r"^<time>[^<]+</time>\nSalut$")
        self.assertEqual(self.saved()[-1]["content"][0]["text"], "Bonjour senpai.")
        self.assertEqual(self.saved()[-2]["content"][0]["text"], "Salut")  # the time is not stored

    def test_history_is_saved_only_after_the_stream_ends(self):
        reply = llm_scr.stream_reply("Salut")
        next(reply)
        self.assertFalse(self.history_file.exists())
        list(reply)
        self.assertTrue(self.history_file.exists())

    def sent(self):
        """The messages Claude received, without the time added to the latest one."""
        return [
            {**message, "content": re.sub(r"^<time>[^<]+</time>\n", "", message["content"])}
            for message in self.request["messages"]
        ]

    def texts(self):
        return [
            (message["role"], llm_scr._message_text(message))
            for message in self.saved()
            if message["role"] != "system"
        ]

    def test_interruption_after_the_reply_was_saved_keeps_only_what_was_heard(self):
        turn = {}
        list(llm_scr.stream_reply("Salut", turn))
        self.assertTrue(turn["saved"])

        llm_scr.save_interrupted_turn("Salut", "Bonjour", turn)

        self.assertEqual(self.texts(), [("user", "Salut"), ("assistant", "Bonjour [interrupted]")])

    def test_interruption_before_the_reply_ends_saves_the_turn_once(self):
        turn = {}
        reply = llm_scr.stream_reply("Salut", turn)
        next(reply)
        llm_scr.save_interrupted_turn("Salut", "", turn)
        list(reply)  # the stream finishing later must not add the full reply

        self.assertEqual(self.texts(), [("user", "Salut"), ("assistant", "[interrupted]")])

    def test_system_prompt_explains_the_interrupted_mark(self):
        self.assertIn(llm_scr.INTERRUPTED_MARK, llm_scr.SYSTEM_PROMPT_TEXT)

    def test_old_gemini_and_openai_history_is_sent_as_text(self):
        old_history = [
            {"role": "system", "content": [{"type": "input_text", "text": "Old prompt"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Salut"}]},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Bonjour"}],
                "gemini_content": {"role": "model", "parts": [{"text": "Bonjour"}]},
                "gemini_model": "gemini-3.8-flash",
            },
        ]
        self.history_file.write_text(json.dumps(old_history), encoding="utf-8")

        answer = llm_scr.llm_response("Comment ça va ?")

        self.assertEqual(answer, "Bonjour senpai.")
        self.assertEqual(
            self.sent(),
            [
                {"role": "user", "content": "Salut"},
                {"role": "assistant", "content": "Bonjour"},
                {"role": "user", "content": "Comment ça va ?"},
            ],
        )
        saved = self.saved()
        self.assertEqual(saved[:3], old_history)
        self.assertEqual(saved[-1], {"role": "assistant", "content": [{"type": "output_text", "text": answer}]})

    def test_refusal_does_not_save_history(self):
        self.stop_reason = "refusal"
        self.stop_details = types.SimpleNamespace(category="cyber")

        with self.assertRaisesRegex(RuntimeError, "declined.*cyber"):
            llm_scr.llm_response("Salut")

        self.assertFalse(self.history_file.exists())

    def test_empty_reply_does_not_change_history(self):
        original = json.dumps(llm_scr.SYSTEM_PROMPT)
        self.history_file.write_text(original, encoding="utf-8")
        self.chunks = []

        with self.assertRaisesRegex(RuntimeError, "returned no text"):
            llm_scr.llm_response("Salut")

        self.assertEqual(self.history_file.read_text(encoding="utf-8"), original)

    def test_reply_cut_at_max_tokens_is_still_saved(self):
        self.stop_reason = "max_tokens"

        with patch("builtins.print") as printed:
            answer = llm_scr.llm_response("Salut")

        self.assertEqual(self.saved()[-1]["content"][0]["text"], answer)
        self.assertIn("max_tokens", printed.call_args.args[0])

    def test_interrupted_history_save_keeps_previous_file(self):
        original = json.dumps(llm_scr.SYSTEM_PROMPT)
        self.history_file.write_text(original, encoding="utf-8")

        def interrupted_dump(_history, history_file, **_kwargs):
            history_file.write('[{"incomplete":')
            raise OSError("interrupted write")

        with patch.object(llm_scr.json, "dump", side_effect=interrupted_dump):
            with self.assertRaisesRegex(OSError, "interrupted write"):
                llm_scr.save_history([{"role": "user", "content": "Salut"}])

        self.assertEqual(self.history_file.read_text(encoding="utf-8"), original)
        self.assertEqual(list(self.history_file.parent.glob(".chat_history.json.*.tmp")), [])

    def test_only_recent_turns_are_sent_but_full_history_is_saved(self):
        history = list(llm_scr.SYSTEM_PROMPT)
        for index in range(5):
            history.append({"role": "user", "content": [{"type": "input_text", "text": f"u{index}"}]})
            history.append({"role": "assistant", "content": [{"type": "output_text", "text": f"a{index}"}]})
        self.history_file.write_text(json.dumps(history), encoding="utf-8")

        with patch.object(llm_scr, "MAX_HISTORY_TURNS", 2):
            llm_scr.llm_response("latest")

        self.assertEqual(
            [message["content"] for message in self.sent()],
            ["u3", "a3", "u4", "a4", "latest"],
        )
        self.assertEqual(len(self.saved()), len(history) + 2)

    def test_corrupt_history_is_moved_aside_and_conversation_continues(self):
        self.history_file.write_text('[{"role": "user", "content": ', encoding="utf-8")

        answer = llm_scr.llm_response("Salut")

        self.assertEqual(answer, "Bonjour senpai.")
        backups = list(self.history_file.parent.glob("chat_history.json.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), '[{"role": "user", "content": ')
        self.assertEqual(self.saved()[-1]["content"][0]["text"], "Bonjour senpai.")

    def test_missing_api_key_fails_at_startup(self):
        with patch.object(llm_scr, "_claude_client", REAL_CLIENT.__wrapped__), \
             patch.dict(llm_scr.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                llm_scr.warm_up()


    def test_system_prompt_asks_for_emotion_tags(self):
        self.assertTrue(llm_scr.EMOTIONS)
        for emotion in llm_scr.EMOTIONS:
            self.assertIn(f"[{emotion}]", llm_scr.SYSTEM_PROMPT_TEXT)

    def test_saved_reply_starts_a_background_memory_update(self):
        llm_scr.llm_response("Salut")

        self.remember.assert_called_once()
        self.assertEqual(self.remember.call_args.args[0], self.saved())

    def test_memory_summary_is_added_to_the_system_prompt(self):
        self.memory_file.write_text(
            json.dumps({"summary": "The user is called Teo.", "summarized_messages": 4}),
            encoding="utf-8",
        )

        llm_scr.llm_response("Salut")

        self.assertTrue(self.request["system"].startswith(llm_scr.SYSTEM_PROMPT_TEXT))
        self.assertIn("The user is called Teo.", self.request["system"])

    @staticmethod
    def conversation(turns):
        history = list(llm_scr.SYSTEM_PROMPT)
        for index in range(turns):
            history.append({"role": "user", "content": [{"type": "input_text", "text": f"u{index}"}]})
            history.append({"role": "assistant", "content": [{"type": "output_text", "text": f"a{index}"}]})
        return history

    def test_memory_waits_for_a_full_batch_outside_the_window(self):
        with patch.object(llm_scr, "MAX_HISTORY_TURNS", 3), patch.object(llm_scr, "MEMORY_BATCH_TURNS", 2):
            self.assertFalse(llm_scr.update_memory(self.conversation(4)))  # 1 turn outside

        self.assertEqual(self.summary_requests, [])
        self.assertFalse(self.memory_file.exists())

    def test_turns_leaving_the_window_are_summarized_once(self):
        self.memory_file.write_text(
            json.dumps({"summary": "Old memory.", "summarized_messages": 2}), encoding="utf-8"
        )
        history = self.conversation(6)  # 12 messages; the next window holds the last 6

        with patch.object(llm_scr, "MAX_HISTORY_TURNS", 3), patch.object(llm_scr, "MEMORY_BATCH_TURNS", 2):
            self.assertTrue(llm_scr.update_memory(history))
            self.assertFalse(llm_scr.update_memory(history))

        self.assertEqual(len(self.summary_requests), 1)
        request = self.summary_requests[0]
        prompt = request["messages"][0]["content"]
        self.assertIn("Old memory.", prompt)
        self.assertIn("User: u1\nRiko: a1\nUser: u2\nRiko: a2", prompt)
        self.assertNotIn("u0", prompt)
        self.assertNotIn("u3", prompt)
        self.assertEqual(request["system"], llm_scr.MEMORY_PROMPT)
        self.assertEqual(
            json.loads(self.memory_file.read_text(encoding="utf-8")),
            {"summary": self.summary_text, "summarized_messages": 6},
        )

    def test_reset_history_starts_the_memory_count_over(self):
        self.memory_file.write_text(
            json.dumps({"summary": "Old memory.", "summarized_messages": 500}), encoding="utf-8"
        )

        with patch.object(llm_scr, "MAX_HISTORY_TURNS", 1), patch.object(llm_scr, "MEMORY_BATCH_TURNS", 1):
            self.assertTrue(llm_scr.update_memory(self.conversation(2)))

        self.assertIn("User: u0", self.summary_requests[0]["messages"][0]["content"])
        self.assertEqual(json.loads(self.memory_file.read_text(encoding="utf-8"))["summarized_messages"], 2)

    def test_declined_summary_keeps_previous_memory(self):
        original = json.dumps({"summary": "Old memory.", "summarized_messages": 0})
        self.memory_file.write_text(original, encoding="utf-8")
        self.summary_stop_reason = "refusal"

        with patch.object(llm_scr, "MAX_HISTORY_TURNS", 1), patch.object(llm_scr, "MEMORY_BATCH_TURNS", 1):
            self.assertFalse(llm_scr.update_memory(self.conversation(3)))

        self.assertEqual(self.memory_file.read_text(encoding="utf-8"), original)

    def test_corrupt_memory_is_moved_aside(self):
        self.memory_file.write_text("{not json", encoding="utf-8")

        self.assertEqual(llm_scr.load_memory(), {"summary": "", "summarized_messages": 0})
        self.assertEqual(len(list(self.memory_file.parent.glob("memory.json.bak-*"))), 1)


if __name__ == "__main__":
    unittest.main()
