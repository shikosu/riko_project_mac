"""Text responses and conversation history for the configured LLM provider."""

import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_FILE = Path(__file__).resolve().parents[3] / "character_config.yaml"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    char_config = yaml.safe_load(config_file)

PROVIDER = char_config.get("provider", "openai").lower()
if PROVIDER not in {"openai", "gemini"}:
    raise ValueError("provider must be 'openai' or 'gemini' in character_config.yaml")

MODEL = char_config["model"]
FALLBACK_MODELS = char_config.get("fallback_models", [])
if not isinstance(FALLBACK_MODELS, list) or not all(
    isinstance(model, str) and model for model in FALLBACK_MODELS
):
    raise ValueError("fallback_models must be a list of model names in character_config.yaml")
MAX_HISTORY_TURNS = char_config.get("max_history_turns", 30)
if not isinstance(MAX_HISTORY_TURNS, int) or MAX_HISTORY_TURNS < 1:
    raise ValueError("max_history_turns must be a positive integer in character_config.yaml")
SYSTEM_PROMPT_TEXT = char_config["presets"]["default"]["system_prompt"]
SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [{"type": "input_text", "text": SYSTEM_PROMPT_TEXT}],
    }
]
history_path = Path(char_config["history_file"])
HISTORY_FILE = history_path if history_path.is_absolute() else CONFIG_FILE.parent / history_path


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
        backup = HISTORY_FILE.with_name(f"{HISTORY_FILE.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        HISTORY_FILE.replace(backup)
        print(f"[HISTORY] Unreadable history moved to {backup}: {exc}")
        return SYSTEM_PROMPT.copy()


def context_window(messages):
    """Current system prompt plus the latest turns; the full history stays on disk."""
    conversation = [message for message in messages if message["role"] in {"user", "assistant"}]
    window = conversation[-(MAX_HISTORY_TURNS * 2 + 1):]
    # Gemini requires the conversation to start with a user turn.
    while window and window[0]["role"] != "user":
        window.pop(0)
    return SYSTEM_PROMPT + window


def save_history(history):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=HISTORY_FILE.parent,
            prefix=f".{HISTORY_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as history_file:
            temp_path = Path(history_file.name)
            json.dump(history, history_file, indent=2, ensure_ascii=False)
            history_file.flush()
            os.fsync(history_file.fileno())
        os.replace(temp_path, HISTORY_FILE)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or char_config.get("OPENAI_API_KEY")
    if not api_key or api_key == "sk-YOURAPIKEY":
        raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI provider.")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to use the Gemini provider.")
    return genai.Client(api_key=api_key)


def _message_text(message):
    """Read the text from both plain and existing OpenAI Responses history."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)


def _gemini_response(messages):
    from google.genai import errors, types

    models = list(dict.fromkeys([MODEL, *FALLBACK_MODELS]))
    for index, model in enumerate(models):
        contents = [
            types.Content.model_validate_json(json.dumps(message["gemini_content"]))
            if message.get("gemini_content") and message.get("gemini_model") == model
            else types.Content(
                role="model" if message["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=_message_text(message))],
            )
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        try:
            response = _gemini_client().models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_TEXT,
                    temperature=1,
                    max_output_tokens=2048,
                ),
            )
        except errors.ServerError as exc:
            if exc.code != 503:
                raise
            if index == len(models) - 1:
                raise RuntimeError(
                    f"Gemini models unavailable (503): {', '.join(models)}. Try again later."
                ) from exc
            print(f"Gemini model {model} unavailable (503); trying {models[index + 1]}.")
            continue

        if not response.text:
            raise RuntimeError("Gemini returned no text; the conversation history was not saved.")
        # Keep the complete model turn so Gemini thought signatures survive restarts.
        model_content = response.candidates[0].content.to_json_dict()
        return response.text, model_content, model


def _openai_response(messages):
    response = _openai_client().responses.create(
        model=MODEL,
        input=[{"role": message["role"], "content": message["content"]} for message in messages],
        temperature=1,
        top_p=1,
        max_output_tokens=2048,
        stream=False,
        text={"format": {"type": "text"}},
    )
    return response.output_text


def llm_response(user_input):
    messages = load_history()
    messages.append({"role": "user", "content": [{"type": "input_text", "text": user_input}]})

    window = context_window(messages)
    if PROVIDER == "gemini":
        answer, model_content, model_used = _gemini_response(window)
    else:
        answer, model_content, model_used = _openai_response(window), None, None

    assistant_message = {"role": "assistant", "content": [{"type": "output_text", "text": answer}]}
    if model_content is not None:
        assistant_message["gemini_content"] = model_content
        assistant_message["gemini_model"] = model_used
    messages.append(assistant_message)
    save_history(messages)
    return answer
