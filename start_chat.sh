#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$project_dir/.env" ]]; then
    set -a
    source "$project_dir/.env"
    set +a
fi

# Each location can be overridden in .env or the shell:
# SOVITS_DIR, SOVITS_PYTHON, SOVITS_CONFIG, GODOT, AVATAR_DIR.
sovits_dir="${SOVITS_DIR:-$project_dir/third_party/GPT-SoVITS}"
sovits_config="${SOVITS_CONFIG:-GPT_SoVITS/configs/tts_infer.yaml}"

riko_python="$project_dir/.venv/bin/python"

avatar_dir="${AVATAR_DIR:-$HOME/riko-avatar}"
godot_project="$avatar_dir/project.godot"


find_sovits_python() {
    local candidate

    # The bundled Linux install first, then the usual conda/miniforge homes on macOS and Linux.
    for candidate in \
        "${SOVITS_PYTHON:-}" \
        "$project_dir/third_party/miniforge3/envs/gpt-sovits/bin/python" \
        "$HOME/miniforge3/envs/gpt-sovits/bin/python" \
        "$HOME/miniconda3/envs/gpt-sovits/bin/python" \
        "$HOME/anaconda3/envs/gpt-sovits/bin/python" \
        "/opt/homebrew/Caskroom/miniforge/base/envs/gpt-sovits/bin/python" \
        "/opt/homebrew/Caskroom/miniconda/base/envs/gpt-sovits/bin/python" \
        "$sovits_dir/.venv/bin/python"
    do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}


find_godot() {
    local candidate

    if [[ -n "${GODOT:-}" ]]; then
        echo "$GODOT"
        return 0
    fi
    if command -v godot >/dev/null 2>&1; then
        command -v godot
        return 0
    fi
    # macOS installs Godot as an app bundle without a command on PATH.
    for candidate in \
        "/Applications/Godot.app/Contents/MacOS/Godot" \
        "$HOME/Applications/Godot.app/Contents/MacOS/Godot"
    do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

api_url="http://127.0.0.1:9880"

api_pid=""
godot_pid=""


cleanup() {
    echo
    echo "Stopping Riko..."

    if [[ -n "$godot_pid" ]]; then
        kill "$godot_pid" 2>/dev/null || true
        wait "$godot_pid" 2>/dev/null || true
    fi

    if [[ -n "$api_pid" ]]; then
        kill "$api_pid" 2>/dev/null || true
        wait "$api_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


api_ready() {
    local spec

    spec="$(
        curl \
            --silent \
            --show-error \
            --fail \
            --max-time 2 \
            "$api_url/openapi.json" \
            2>/dev/null
    )" || return 1

    [[ "$spec" == *'"/tts"'* ]]
}


# ============================================================
# CHECK RIKO
# ============================================================

if [[ ! -x "$riko_python" ]]; then
    echo "Riko's .venv is missing." >&2
    exit 1
fi


# ============================================================
# CHECK GPT-SOVITS
# ============================================================

sovits_python="$(find_sovits_python)" || sovits_python=""

if [[ -z "$sovits_python" || ! -f "$sovits_dir/api_v2.py" ]]; then
    echo "GPT-SoVITS is missing." >&2
    echo "Expected api_v2.py in: $sovits_dir (set SOVITS_DIR)" >&2
    echo "and a 'gpt-sovits' conda environment (set SOVITS_PYTHON to its python)." >&2
    exit 1
fi


# ============================================================
# START GPT-SOVITS
# ============================================================

if ! api_ready; then

    mkdir -p "$project_dir/audio"

    (
        cd "$sovits_dir"

        exec "$sovits_python" \
            api_v2.py \
            -a 127.0.0.1 \
            -p 9880 \
            -c "$sovits_config"

    ) > "$project_dir/audio/sovits.log" 2>&1 &

    api_pid=$!

    echo "Starting GPT-SoVITS API..."


    for ((attempt = 0; attempt < 150; attempt++)); do

        if api_ready; then
            break
        fi

        if ! kill -0 "$api_pid" 2>/dev/null; then

            echo "GPT-SoVITS failed to start." >&2
            echo "See audio/sovits.log:" >&2

            tail -n 25 \
                "$project_dir/audio/sovits.log" >&2

            exit 1
        fi

        sleep 2
    done


    if ! api_ready; then
        echo "GPT-SoVITS did not become ready." >&2
        echo "See audio/sovits.log." >&2
        exit 1
    fi
fi


echo "GPT-SoVITS API ready."


# ============================================================
# START GODOT AVATAR
# ============================================================

if [[ -f "$godot_project" ]] && godot_bin="$(find_godot)"; then

    echo "Starting Riko avatar..."

    "$godot_bin" \
        --path "$avatar_dir" \
        --editor \
        "$godot_project" &

    godot_pid=$!

elif [[ -f "$godot_project" ]]; then

    echo "WARNING: Godot not found; set GODOT to its executable."

else

    echo "WARNING: Godot avatar project not found:"
    echo "$godot_project"

fi


# ============================================================
# START RIKO
# ============================================================

echo "Starting Riko brain..."

cd "$project_dir"

"$riko_python" server/main_chat.py
