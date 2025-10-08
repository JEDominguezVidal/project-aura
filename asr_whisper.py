"""
Wrapper for executing Whisper and saving transcript to plain text.

This module provides a simple interface to OpenAI's Whisper ASR system,
handling model loading, transcription, and text output. It uses the
`openai-whisper` package for local transcription without requiring
API calls.
"""
from pathlib import Path
import whisper
import logging
from typing import Dict, Any
from utils import setup_logger


def transcribe_whisper(logger: logging.Logger, wav_path: str, out_txt: str, model_name: str = "medium", language: str = "es") -> Dict[str, Any]:
    """
    Transcribe audio file using OpenAI Whisper and save result to text file.

    This function loads the specified Whisper model, performs transcription
    on the input audio file, and saves the resulting text to the output file.
    The transcription includes automatic punctuation and formatting.

    Args:
        logger: Logger instance for logging messages
        wav_path: Path to input WAV file (must be Whisper-compatible format)
        out_txt: Path where transcript text will be saved
        model_name: Whisper model to use (e.g. "tiny", "base", "small", "medium", "large-v3")
        language: Language code for transcription (e.g. "es", "en", "fr")

    Returns:
        Dict[str, Any]: Complete Whisper result dictionary containing:
            - text: Transcribed text
            - segments: Word/phrase level timing (if available)
            - language: Detected language

    Raises:
        FileNotFoundError: If input WAV file doesn't exist
        RuntimeError: If model loading or transcription fails

    Example:
        >>> result = transcribe_whisper('audio.wav', 'transcript.txt', 'large-v3', 'es')
        >>> print(result['text'])
        'Hi, this is just a test.'
    """
    logger = logger

    # Verify input file exists
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

    # Load Whisper model (this can take some time for larger models)
    try:
        logger.info("Loading Whisper model: %s", model_name)
        model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully: %s", model_name)
    except Exception as e:
        logger.error("Failed to load Whisper model %s: %s", model_name, e)
        raise RuntimeError(f"Model loading failed: {e}")

    # Perform transcription with specified language
    # fp16=False ensures compatibility across different hardware
    logger.info("Starting transcription of: %s", wav_path)
    try:
        result = model.transcribe(wav_path, language=language, fp16=False)
    except Exception as e:
        logger.error("Transcription failed for %s: %s", wav_path, e)
        raise RuntimeError(f"Transcription failed: {e}")

    # Extract and clean the transcribed text
    text = result.get("text", "").strip()

    # Save transcript to output file
    try:
        Path(out_txt).write_text(text, encoding="utf-8")
        logger.info("Transcription saved to: %s", out_txt)
        logger.info("Transcribed text length: %d characters", len(text))
    except Exception as e:
        logger.error("Failed to save transcript to %s: %s", out_txt, e)
        raise RuntimeError(f"Failed to save transcript: {e}")

    return result
