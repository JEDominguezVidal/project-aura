#!/usr/bin/env python3
"""
Main CLI entry point that orchestrates the complete Whisper+MFA dataset building pipeline.

This module serves as the primary interface for the audio processing pipeline that:
1. Preprocesses audio files for Whisper compatibility
2. Performs automatic speech recognition using OpenAI Whisper
3. Runs forced alignment using Montreal Forced Aligner (MFA)
4. Parses alignment results to extract sentence-level timestamps
5. Generates segmented audio clips with corresponding transcriptions
6. (Optional) Generates TTS training dataset CSV, when --generate-dataset is used

The pipeline is designed to create training datasets from long-form audio recordings
by breaking them into sentence-level segments with precise timing information.
"""
import argparse
import logging
from pathlib import Path

from core.utils import setup_logger
from core.audio_preprocess import ensure_wav_for_whisper
from core.asr_whisper import transcribe_whisper
from core.align_mfa import run_mfa_alignment, parse_textgrid_for_sentences
from core.segmenter import export_sentence_clips
from core.generate_training_dataset import generate_tts_dataset
from core.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_MFA_LANG,
    DEFAULT_OUTPUT_FREQ,
    DEFAULT_OUTPUT_DIR
)


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
            - outfreq: Output sample rate in Hz for clips
    """
    parser = argparse.ArgumentParser(description="Whisper+MFA dataset builder")
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE, help=f"Language code (default: {DEFAULT_LANGUAGE})")
    parser.add_argument("--whisper_model", default=DEFAULT_WHISPER_MODEL, help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--mfa_lang", default=DEFAULT_MFA_LANG, help=f"MFA language model (default: {DEFAULT_MFA_LANG})")
    parser.add_argument("--outfreq", type=int, default=DEFAULT_OUTPUT_FREQ, help=f"Output sample rate in Hz (default: {DEFAULT_OUTPUT_FREQ})")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate TTS training dataset CSV from clips")
    return parser.parse_args()


def main() -> None:
    """
    Execute the complete audio processing pipeline.

    This function orchestrates the entire process:
    1. Preprocesses audio for Whisper compatibility
    2. Transcribes audio using Whisper ASR
    3. Performs forced alignment with MFA
    4. Extracts sentence-level timestamps
    5. Generates segmented audio clips
    6. Generates TTS training dataset CSV (optional, when --generate-dataset is used)

    The function will exit with an error code if any step fails.
    """
    # Parse command line arguments
    args = parse_args()

    # Initialise logging system
    logger = setup_logger()

    # Create output directory structure
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocess audio for Whisper compatibility
    logger.info("Preprocessing audio for Whisper...")
    preproc_wav = outdir / "input_16k_mono.wav"
    ensure_wav_for_whisper(Path(args.input), preproc_wav)

    # Step 2: Perform ASR with Whisper
    logger.info("Transcribing with Whisper (%s)...", args.whisper_model)
    transcript_txt = outdir / "transcript.txt"
    transcribe_whisper(str(preproc_wav), str(transcript_txt), model_name=args.whisper_model, language=args.lang)

    # Step 3: Run MFA alignment
    logger.info("Running MFA for alignment...")
    mfa_output_dir = outdir / "mfa_output"
    mfa_output_dir.mkdir(exist_ok=True)
    textgrid_path = run_mfa_alignment(wav_path=preproc_wav, transcript_path=transcript_txt, out_dir=mfa_output_dir, mfa_lang=args.mfa_lang)

    # Step 4: Parse TextGrid and generate sentence timestamps
    logger.info("Parsing TextGrid to extract sentences with timestamps...")
    if textgrid_path is None:
        logger.error("No TextGrid found in %s", mfa_output_dir)
        raise SystemExit(1)

    sentences = parse_textgrid_for_sentences(textgrid_path, transcript_txt)

    # Step 5: Export clips per sentence
    clips_dir = outdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    generated_clips = export_sentence_clips(preproc_wav, sentences, clips_dir, outfreq=args.outfreq)

    # Step 6: Generate TTS training dataset (optional)
    if args.generate_dataset:
        logger.info("Generating TTS training dataset...")
        success = generate_tts_dataset(clips_dir, outdir)
        if success:
            dataset_dir = outdir / "dataset"
            logger.info("TTS dataset created successfully in: %s", dataset_dir)
        else:
            logger.warning("Failed to generate TTS dataset")

    logger.info("Process completed. Clips generated in: %s", clips_dir)


if __name__ == "__main__":
    main()
