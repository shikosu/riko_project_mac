# Project Riko

Project Riko is an anime focused LLM project by Just Rayen. She listens and remembers your conversations. It combines Claude, GPT-SoVITS and Piper voice synthesis, and Faster-Whisper ASR into a configurable conversational pipeline. She answers in French when you speak French and in English otherwise.

**tested with python 3.10 on Windows >10, Linux Ubuntu and macOS (Apple Silicon)**
## ✨ Features

- 💬 **LLM-based dialogue** using Claude (configurable system prompts)
- 🇫🇷 **French and English**, detected locally with no extra API calls
- 🧠 **Conversation memory**: recent turns word for word, older ones as a long-term summary
- 😊 **Facial expressions**: Claude picks an emotion that the avatar shows while she speaks
- 🔊 **Voice generation** via GPT-SoVITS (English) and Piper (French), spoken sentence by sentence while Claude is still writing
- 🎧 **Speech recognition** using Faster-Whisper
- 📁 Clean YAML-based config for personality configuration


## ⚙️ Configuration

Prompts and model settings are stored in `character_config.yaml`:

```yaml
history_file: chat_history.json
model: "claude-sonnet-5-5"
effort: low
max_tokens: 8000
max_history_turns: 30
presets:
  default:
    system_prompt: |
      You are a helpful assistant named Riko.
      ...
      Reply in the language of the user's latest message: French if they wrote in French, otherwise English.

asr:
  whisper_model: small
  device: cpu
  compute_type: int8

sovits_ping_config:
  text_lang: en
  prompt_lang : en
  ref_audio_path : character_files/riko_reference.wav
  prompt_text : How's everyone doing tonight? ...
  speed_params:
    text_split_method: cut0
    batch_size: 1
    parallel_infer: true

piper_config:
  model_path: character_files/piper/fr_FR-siwis-medium.onnx
```

You can define personalities by modifying the system prompt. Keep its last lines: they make Riko answer in your language with short, plain spoken sentences, which the voice reads well. Set `ref_audio_path` to an absolute path or a path relative to the project root. GPT-SoVITS only accepts a reference clip of 3 to 10 seconds, and `prompt_text` must be the exact words spoken in it; Riko sends the resolved absolute path to GPT-SoVITS.

The full conversation is kept in `history_file`, but only the current system prompt and the last `max_history_turns` exchanges are sent to Claude, so requests stay fast and cheap. Only the text of each reply is stored. Older Gemini or OpenAI history files are still readable. If the history file cannot be read, it is renamed to `chat_history.json.bak-<timestamp>` and a new conversation starts. A reply is saved only once Claude has finished it.

### Claude API key

Create an API key in the [Claude Console](https://platform.claude.com/settings/keys), then put it in a `.env` file at the project root (it is ignored by git and loaded by `start_chat.sh`):

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Or export it in the shell before starting Riko. On Windows PowerShell, use `$env:ANTHROPIC_API_KEY = "sk-ant-..."`. Do not put a real API key in the tracked YAML file.

If a Claude safety classifier declines a request, Riko asks the API to retry it on another Claude model automatically (`fallbacks: "default"`). A reply that is still declined is not saved to history.

### French and English

Riko works out your language locally, without calling the API:

* **Voice:** Faster-Whisper detects French or English before transcribing, so each recording is transcribed once. Use a multilingual model (`tiny`, `base`, `small`, ...): the `.en` models cannot hear French.
* **Typed text:** `fast_langdetect` guesses the language of each line.

Claude answers in the same language. English replies use Riko's GPT-SoVITS voice. GPT-SoVITS cannot speak French, so French replies use [Piper](https://github.com/OHF-Voice/piper1-gpl), a fast local voice. Download the French voice once:

```bash
.venv/bin/python -m piper.download_voices --download-dir character_files/piper fr_FR-siwis-medium
```

Without it, Riko still answers in French in the terminal but cannot say it aloud.

### Long-term memory

Claude sees the last `max_history_turns` exchanges word for word. When older exchanges leave that window, Riko summarizes them in the background, without delaying her reply, into `memory_file` (`memory.json` by default), and the summary is added to her system prompt. To keep API use low, this happens once per `memory_batch_turns` exchanges (10 by default), in one small request. Set `memory_model` to use another model for summaries. Delete `memory.json` to make Riko forget.

### Facial expressions

Claude starts each reply, and each change of mood, with a tag such as `[happy]`. Riko removes the tags before speaking and sends `{"type": "emotion", "value": "happy"}` to the avatar just before the matching sentence is played. The allowed names are set by `emotions` in `character_config.yaml` (by default the VRM presets: `neutral`, `happy`, `angry`, `sad`, `relaxed`, `surprised`) and must match the expressions your Godot project handles. Set `emotions: []` to turn this off.

### Response speed

Riko does not wait for the full reply. As soon as Claude finishes a sentence, it is synthesized, and lip sync runs in its own thread, so the next sentence is ready before the current one ends. `[TIMING]` lines in the terminal show each step: transcription, Claude's first token, speech synthesis, lip sync, the first audio and any gap between sentences. `tests/benchmark_voice.py` times synthesis and lip sync on fixed sentences. To tune speed:

* `effort`: how much Claude thinks first. `low` is fastest; `medium` or `high` give more considered answers.
* `streaming_mode` (in `sovits_ping_config`): the first English sentence is streamed from GPT-SoVITS, so Riko starts talking ~0.5 s after Claude writes it instead of ~2.5 s. `3` starts fastest, `2` sounds better but starts later, `0` turns it off. Later sentences are synthesized while she speaks and are not streamed.
* `speed_params`: extra GPT-SoVITS `/tts` parameters, sent as-is. On Apple Silicon, GPT-SoVITS v2 is faster on the CPU than on the GPU (`mps`).
* `lipsync`: `energy` (default) moves the mouth with the loudness of the voice and takes a few milliseconds; `rhubarb` reads phonemes for finer shapes but adds about 0.3 s per second of audio. Rhubarb receives each sentence's text and uses its fast `phonetic` recognizer.
* `fillers`: short sounds ("Hmm...", "Hein ?") Riko says in voice mode when her answer takes more than 0.4 s. They are synthesized once into `audio/fillers`; delete that folder after changing her voice. Set `fillers_enabled: false` to turn them off.
* `whisper_model` and `language_model`: `language_model` (`base`) detects French or English, then `whisper_model` (`small`) transcribes once in that language. `cpu_threads: 8` suits Apple M-series CPUs.


## 🛠️ Setup

### Install Dependencies

With [uv](https://docs.astral.sh/uv/) installed, run from the project root:

```bash
./install_reqs.sh
```

It creates `.venv` and downloads the French Piper voice. On Linux it also installs PyTorch for CUDA 12.6 and the GPT-SoVITS requirements. On macOS it installs only what Riko needs (`extra-req.txt`): Riko uses no PyTorch, and GPT-SoVITS runs in its own environment. It warns if `ffmpeg` or `rhubarb` is missing.

To install by hand instead:

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r extra-req.txt                     # macOS
uv pip install -r extra-req.txt -r requirements.txt # Linux / Windows
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

**If you want to use GPU support for Faster whisper** Make sure you also have:

* CUDA & cuDNN installed correctly (for Faster-Whisper GPU support)
* `ffmpeg` installed (for audio processing)


## 🧪 Usage

On Linux or macOS, set `ANTHROPIC_API_KEY` as described above, then run `./start_chat.sh` from the project root. It starts the local GPT-SoVITS API and Riko, and opens the separate Godot avatar project at `~/riko-avatar` when present. The API log is written to `audio/sovits.log`. Install the Rhubarb executable on `PATH` for avatar lip sync; voice playback still works without it.

`start_chat.sh` looks for GPT-SoVITS in `third_party/GPT-SoVITS`, and for its Python in a conda environment named `gpt-sovits` (in `third_party/miniforge3`, `~/miniforge3`, `~/miniconda3`, `~/anaconda3`, or Homebrew's miniforge/miniconda). It finds Godot as a `godot` command or in `/Applications/Godot.app`. Override any of them in `.env`:

```bash
SOVITS_DIR=/path/to/GPT-SoVITS
SOVITS_PYTHON=/path/to/envs/gpt-sovits/bin/python
SOVITS_CONFIG=GPT_SoVITS/configs/tts_infer.yaml
GODOT=/Applications/Godot.app/Contents/MacOS/Godot
AVATAR_DIR=$HOME/riko-avatar
```

**On macOS:**

* Allow your terminal to use the microphone (System Settings → Privacy & Security → Microphone) the first time Riko records.
* There is no CUDA on a Mac. If GPT-SoVITS fails to start with a CUDA error, set `device: cpu` and `is_half: false` in its `tts_infer.yaml` (or point `SOVITS_CONFIG` to a copy with those values).

The steps below start the two processes separately.

### 1. Launch the GPT-SoVITS API

If GPT-SoVITS is not installed, follow its [official Linux installation instructions](https://github.com/RVC-Boss/GPT-SoVITS?tab=readme-ov-file#linux) first. In a separate terminal, start its API from the GPT-SoVITS directory after configuring its models and weights:

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

Riko calls its `/tts` endpoint. If the API is unavailable, Riko prints the reply and continues without voice playback.

### 2. Run the main script:

```bash
.venv/bin/python server/main_chat.py
```

Run this from the project root. On Windows PowerShell, use `.venv\Scripts\python.exe server/main_chat.py`. If the virtual environment is activated, `python server/main_chat.py` also works.

The flow:

1. Riko listens to your voice via microphone (push to talk)
2. Transcribes it with Faster-Whisper
3. Detects French or English
4. Streams Claude's reply (with history)
5. Synthesizes each sentence with GPT-SoVITS (English) or Piper (French)
6. Plays each sentence as soon as it is ready

Press ENTER while Riko is speaking to cut her off: she stops at once and recording starts, so you can talk right away. Her history keeps only what she said before you interrupted.

To type instead of speaking, press `T` at the "Press ENTER to start recording, or T to type..." prompt. Each line you type is sent to Riko, who still answers with her voice. Type `/v` or press Ctrl-D to go back to voice mode.


## 📌 TODO / Future Improvements

* [ ] GUI or web interface
* [ ] Live microphone input support
* [x] Avatar facial expressions chosen by the LLM
* [ ] Emotion or tone control in speech synthesis
* [ ] VRM model frontend


## 🧑‍🎤 Credits

* Voice synthesis powered by [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* ASR via [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* French voice via [Piper](https://github.com/OHF-Voice/piper1-gpl)
* Language model via [Claude](https://platform.claude.com/docs)


## 📜 License

MIT — feel free to clone, modify, and build your own waifu voice companion.
