"""
Generate WAV clips per sentence using ffmpeg and save transcriptions to text files.
"""
from pathlib import Path


def export_sentence_clips(source_wav: Path, sentences: list, out_dir: Path, min_dur: float = 0.8, pre_roll: float = 0.2, post_roll: float = 0.2) -> list:
    pass
