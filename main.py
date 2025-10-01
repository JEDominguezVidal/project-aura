#!/usr/bin/env python3
"""
Main CLI entry point that orchestrates the complete Whisper+MFA dataset building pipeline.

The pipeline is designed to create training datasets from long-form audio recordings
by breaking them into sentence-level segments with precise timing information.
"""
import argparse
from pathlib import Path

from utils import setup_logger
from audio_preprocess import ensure_wav_for_whisper
from asr_whisper import transcribe_whisper
from align_mfa import run_mfa_alignment, parse_textgrid_for_sentences
from segmenter import export_sentence_clips


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the dataset building pipeline.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - input: Path to input WAV file
            - outdir: Output directory for processed files
            - lang: Language code for transcription
            - whisper_model: Whisper model to use for ASR
            - mfa_lang: Language model for MFA alignment
    """
    parser = argparse.ArgumentParser(description="Whisper+MFA dataset builder")
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--outdir", default="./output", help="Output directory")
    parser.add_argument("--lang", default="es", help="Language code (e.g. es, en)")
    parser.add_argument("--whisper_model", default="large-v3", help="Whisper model to use")
    parser.add_argument("--mfa_lang", default="spanish", help="Language model name for MFA")
    return parser.parse_args()


def main() -> None:
    """
    Execute the complete audio processing pipeline.
    """
    # Parse command line arguments
    args = parse_args()


if __name__ == "__main__":
    main()
