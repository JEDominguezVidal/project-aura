"""
Wrapper for executing Whisper and saving transcript to plain text.
"""
import whisper


def transcribe_whisper(wav_path: str, out_txt: str, model_name: str = "large-v3", language: str = "es") -> None:
    pass