"""
Audio preprocessing functions for Whisper compatibility.

This module handles audio format conversion to ensure compatibility with
OpenAI Whisper's requirements. It converts various audio formats to the
specific format expected by Whisper for optimal performance.

Whisper requires:
- WAV format
- Mono audio (1 channel)
- 16kHz sample rate
- 16-bit PCM encoding
"""
import subprocess
from pathlib import Path
import logging
from utils import setup_logger


def ensure_wav_for_whisper(logger: logging.Logger, input_path: Path, output_path: Path) -> None:
    """
    Convert audio to WAV mono 16kHz 16-bit PCM format using ffmpeg.

    This function ensures that audio files are in the exact format required
    by Whisper for transcription. It converts sample rate, channel count,
    and encoding to match Whisper's expectations.

    Args:
        logger: Logger instance for logging messages
        input_path: Path to input audio file (any format supported by ffmpeg)
        output_path: Path where the converted WAV file will be saved

    Raises:
        subprocess.CalledProcessError: If ffmpeg conversion fails
        FileNotFoundError: If input file doesn't exist

    Example:
        >>> ensure_wav_for_whisper(Path('input.mp3'), Path('output.wav'))
    """
    logger = logger

    # Ensure input and output are Path objects for consistent handling
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Verify input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Ensure parent directory for output file exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct ffmpeg command for Whisper-compatible conversion
    # -y: Overwrite output file if it exists
    # -i: Input file path
    # -ac 1: Convert to mono (1 channel)
    # -ar 16000: Resample to 16kHz
    # -sample_fmt s16: Use 16-bit PCM encoding
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",           # Convert to mono
        "-ar", "16000",       # Resample to 16kHz
        "-sample_fmt", "s16", # 16-bit PCM encoding
        str(output_path)
    ]

    logger.debug("Running ffmpeg: %s", " ".join(cmd))

    # Execute ffmpeg conversion
    try:
        subprocess.run(cmd, check=True)
        logger.info("Audio conversion completed: %s -> %s", input_path, output_path)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg conversion failed with return code %d", e.returncode)
        raise
