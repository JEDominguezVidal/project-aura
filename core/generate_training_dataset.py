"""
Generate training dataset CSV for TTS fine-tuning.

This module creates CSV files mapping audio clips to their corresponding text
transcripts, suitable for training TTS models like Orpheus or other
voice cloning systems that require filename,text format datasets.
"""
from pathlib import Path
import csv
import logging


def generate_tts_dataset(logger: logging.Logger, clips_dir: Path, output_csv: Path, relative_paths=True):
    """
    Generates a CSV file mapping .wav files to their corresponding text transcripts.

    Args:
        logger: Logger instance for logging messages
        clips_dir (Path): Directory containing .wav and .txt files.
        output_csv (Path): Output CSV file path (e.g. dataset.csv).
        relative_paths (bool): If True, stores only the filename (e.g. '0001.wav'),
                               otherwise stores absolute paths.

    Output CSV format:
        filename,text
        0001.wav,Hello there!
        0002.wav,I am very tired.
    """
    logger = logger
    clips_dir = Path(clips_dir)
    wav_files = sorted(clips_dir.glob("*.wav"))

    if not wav_files:
        logger.warning("No .wav files found in %s", clips_dir)
        return

    logger.info("Generating TTS dataset CSV from %d clips...", len(wav_files))

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "text"])

        count = 0
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                logger.warning("No transcription found for %s, skipping", wav_path.name)
                continue

            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                logger.warning("Empty transcription in %s, skipping", txt_path)
                continue

            filename = wav_path.name if relative_paths else str(wav_path.resolve())
            writer.writerow([filename, text])
            count += 1

    logger.info("Dataset CSV written to %s with %d entries.", output_csv, count)
    if count == 0:
        logger.warning("No valid pairs were found; CSV is empty.")
        return False
    return True
