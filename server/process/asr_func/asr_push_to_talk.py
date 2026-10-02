import os
import sys

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


def record_and_transcribe(model, output_file="recording.wav", samplerate=44100, wait_for_start=True):
    """
    Simple push-to-talk recorder: record -> save -> transcribe -> return text
    """
    
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    if wait_for_start:
        print("Press ENTER to start recording...")
        input()
    
    print("🔴 Recording... Press ENTER to stop")
    
    # Record audio directly
    recording = sd.rec(int(60 * samplerate), samplerate=samplerate, channels=1, dtype='float64')
    input()  # Wait for stop
    sd.stop()
    
    print("⏹️  Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)
    
    print("🎯 Transcribing...")
    
    # Transcribe
    segments, _ = model.transcribe(output_file)
    transcription = " ".join([segment.text for segment in segments])
    
    print(f"Transcription: {transcription}")
    return transcription.strip()


# Example usage
if __name__ == "__main__":
    model = WhisperModel("base.en", device="cpu", compute_type="float32")
    result = record_and_transcribe(model)
    print(f"Got: '{result}'")
    