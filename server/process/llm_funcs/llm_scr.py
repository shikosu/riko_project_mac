"""Claude text responses and conversation history."""

import json
import os
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_FILE = Path(__file__).resolve().parents[3] / "character_config.yaml"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    char_config = yaml.safe_load(config_file)

MODEL = char_config.get("model", "claude-sonnet-5-5")
EFFORT = char_config.get("effort", "low")
if EFFORT not in {"low", "medium", "high", "xhigh", "max"}:
    raise ValueError("effort must be low, medium, high, xhigh or max in character_config.yaml")
MAX_TOKENS = char_config.get("max_tokens", 8000)
if not isinstance(MAX_TOKENS, int) or MAX_TOKENS < 1:
    raise ValueError("max_tokens must be a positive integer in character_config.yaml")
# Retries the request on another Claude model when a safety classifier declines it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_HISTORY_TURNS = char_config.get("max_history_turns", 30)
if not isinstance(MAX_HISTORY_TURNS, int) or MAX_HISTORY_TURNS < 1:
    raise ValueError("max_history_turns must be a positive integer in character_config.yaml")
EMOTIONS = [emotion.lower() for emotion in char_config.get("emotions", [])]
EMOTION_RULE = (
    "\n\nStart each reply with your facial expression in square brackets, one of: "
    + ", ".join(f"[{emotion}]" for emotion in EMOTIONS)
    + ". When your mood changes later in the reply, put a new tag before that sentence. "
    "Use only those exact tags, with no other words inside the brackets. "
    "The tags move your avatar's face and are never read aloud."
)
INTERRUPTED_MARK = "[interrupted]"
INTERRUPTED_RULE = (
    f"\n\nIf one of your earlier replies ends with {INTERRUPTED_MARK}, the user cut you off there "
    "and only heard the words before it. Never write that mark yourself."
)
SYSTEM_PROMPT_TEXT = char_config["presets"]["default"]["system_prompt"].rstrip() + (
    EMOTION_RULE if EMOTIONS else ""
) + INTERRUPTED_RULE
SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [{"type": "input_text", "text": SYSTEM_PROMPT_TEXT}],
    }
]


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else CONFIG_FILE.parent / path


HISTORY_FILE = _project_path(char_config["history_file"])
MEMORY_FILE = _project_path(char_config.get("memory_file", "memory.json"))
MEMORY_BATCH_TURNS = char_config.get("memory_batch_turns", 10)
if not isinstance(MEMORY_BATCH_TURNS, int) or MEMORY_BATCH_TURNS < 1:
    raise ValueError("memory_batch_turns must be a positive integer in character_config.yaml")
MEMORY_MODEL = char_config.get("memory_model", MODEL)
MEMORY_PROMPT = """You maintain the long-term memory of Riko, a voice companion, about the person she talks with.
You receive her current memory and an older part of the conversation that is leaving her short-term context.
Rewrite the memory so it also keeps what matters from that conversation: facts about the person (name, life, \
tastes, plans), important events, promises, and running jokes or how the relationship feels.
Drop small talk and anything already covered. Keep it under 300 words, as short plain-text lines in English.
Reply with the updated memory only."""
_memory_lock = threading.Lock()
_history_lock = threading.Lock()  # a finished reply and an interruption may save at the same time


def load_history():
    if not HISTORY_FILE.exists():
        return SYSTEM_PROMPT.copy()
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as history_file:
            history = json.load(history_file)
        if not isinstance(history, list) or not all(
            isinstance(message, dict) and "role" in message for message in history
        ):
            raise ValueError("history must be a list of messages")
        return history
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        # Keep the unreadable file for inspection and start a fresh conversation.
        _move_aside(HISTORY_FILE, "history", exc)
        return SYSTEM_PROMPT.copy()


def _move_aside(path, label, exc):
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    path.replace(backup)
    print(f"[{label.upper()}] Unreadable {label} moved to {backup}: {exc}")


def context_window(messages):
    """Current system prompt plus the latest turns; the full history stays on disk."""
    conversation = [message for message in messages if message["role"] in {"user", "assistant"}]
    window = conversation[-(MAX_HISTORY_TURNS * 2 + 1):]
    # Claude requires the conversation to start with a user turn.
    while window and window[0]["role"] != "user":
        window.pop(0)
    return SYSTEM_PROMPT + window


def _write_json_atomically(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as json_file:
            temp_path = Path(json_file.name)
            json.dump(data, json_file, indent=2, ensure_ascii=False)
            json_file.flush()
            os.fsync(json_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_history(history):
    _write_json_atomically(HISTORY_FILE, history)


def load_memory():
    """Long-term summary of the turns that no longer fit in the context window."""
    empty = {"summary": "", "summarized_messages": 0}
    if not MEMORY_FILE.exists():
        return empty
    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as memory_file:
            memory = json.load(memory_file)
        if not (
            isinstance(memory, dict)
            and isinstance(memory.get("summary"), str)
            and isinstance(memory.get("summarized_messages"), int)
        ):
            raise ValueError("memory must hold a summary and a summarized_messages count")
        return memory
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        _move_aside(MEMORY_FILE, "memory", exc)
        return empty


def _system_prompt(memory):
    if not memory["summary"]:
        return SYSTEM_PROMPT_TEXT
    return (
        f"{SYSTEM_PROMPT_TEXT}\n\n"
        f"What you remember from earlier conversations with the user:\n{memory['summary']}"
    )


def _message_text(message):
    """Read the text from plain and structured (older OpenAI/Gemini) history."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)


@lru_cache(maxsize=1)
def _claude_client():
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError("Set ANTHROPIC_API_KEY (for example in .env) to use Claude.")
    return anthropic.Anthropic(max_retries=3, timeout=60.0)


def warm_up():
    """Create the client at startup so a missing key fails before the first turn."""
    _claude_client()


def _claude_messages(messages):
    # Only text is replayed: the window drops old turns, which would invalidate thinking blocks.
    return [
        {"role": message["role"], "content": _message_text(message)}
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]


def _with_time(messages):
    """Tell Claude the current time in the latest message only, so earlier turns stay cached."""
    now = time.strftime("%A %d %B %Y, %H:%M")
    latest = messages[-1]
    return messages[:-1] + [{**latest, "content": f"<time>{now}</time>\n{latest['content']}"}]


def stream_reply(user_input, turn=None):
    """Yield Claude's reply as text chunks; save the turn once the reply is complete.

    turn, a dict shared with save_interrupted_turn, records whether this turn was saved.
    """
    turn = {} if turn is None else turn
    messages = load_history()
    messages.append({"role": "user", "content": [{"type": "input_text", "text": user_input}]})

    chunks = []
    with _claude_client().beta.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(load_memory()),
        messages=_with_time(_claude_messages(context_window(messages))),
        output_config={"effort": EFFORT},
        cache_control={"type": "ephemeral"},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            yield text
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        category = final.stop_details.category if final.stop_details else None
        raise RuntimeError(f"Claude declined to answer ({category}); the history was not saved.")
    answer = "".join(chunks).strip()
    if not answer:
        raise RuntimeError("Claude returned no text; the history was not saved.")
    if final.stop_reason == "max_tokens":
        print(f"[LLM] Reply cut off at max_tokens={MAX_TOKENS}; raise it in character_config.yaml.")

    messages.append({"role": "assistant", "content": [{"type": "output_text", "text": answer}]})
    with _history_lock:
        if turn.get("interrupted"):
            return  # save_interrupted_turn already kept what the user heard
        save_history(messages)
        turn["saved"] = True
    remember_in_background(messages)


def save_interrupted_turn(user_input, spoken, turn):
    """Keep only the part of the reply the user heard before cutting Riko off."""
    answer = f"{spoken} {INTERRUPTED_MARK}".strip()
    with _history_lock:
        turn["interrupted"] = True
        messages = load_history()
        if turn.get("saved") and messages and messages[-1]["role"] == "assistant":
            messages.pop()  # the full reply, which she did not finish saying
        else:
            messages.append({"role": "user", "content": [{"type": "input_text", "text": user_input}]})
        messages.append({"role": "assistant", "content": [{"type": "output_text", "text": answer}]})
        save_history(messages)
    print(f"[LLM] Reply cut off; saved what was said: {answer}")


def llm_response(user_input):
    return "".join(stream_reply(user_input)).strip()


def update_memory(messages):
    """Fold turns that left the context window into the summary; return True if it changed."""
    conversation = [message for message in messages if message["role"] in {"user", "assistant"}]
    with _memory_lock:
        memory = load_memory()
        done = memory["summarized_messages"]
        if done > len(conversation):
            done = 0  # the history was reset; the summary stays, the count starts over
        # Messages before this index are no longer sent with the next request.
        outside = len(conversation) - MAX_HISTORY_TURNS * 2
        if outside - done < MEMORY_BATCH_TURNS * 2:
            return False

        excerpt = "\n".join(
            f"{'User' if message['role'] == 'user' else 'Riko'}: {_message_text(message)}"
            for message in conversation[done:outside]
        )
        response = _claude_client().beta.messages.create(
            model=MEMORY_MODEL,
            max_tokens=4000,
            system=MEMORY_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"<memory>\n{memory['summary'] or '(empty)'}\n</memory>\n\n"
                    f"<conversation>\n{excerpt}\n</conversation>"
                ),
            }],
            output_config={"effort": "low"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        summary = "".join(block.text for block in response.content if block.type == "text").strip()
        if response.stop_reason == "refusal" or not summary:
            print("[MEMORY] Summary not updated; Claude returned no summary.")
            return False

        _write_json_atomically(MEMORY_FILE, {"summary": summary, "summarized_messages": outside})
        print(f"[MEMORY] Summarized {outside - done} older messages.")
        return True


def remember_in_background(messages):
    """Update the summary without delaying the spoken reply."""
    def run():
        try:
            update_memory(messages)
        except Exception as exc:
            print(f"[MEMORY ERROR] {exc}")

    threading.Thread(target=run, daemon=True, name="RikoMemory").start()
