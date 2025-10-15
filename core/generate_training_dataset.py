"""
Generate training dataset CSV for TTS fine-tuning.

This module creates CSV files mapping audio clips to their corresponding text
transcripts, suitable for training TTS models like Orpheus or other
voice cloning systems that require filename,text format datasets.
"""
from pathlib import Path
import csv
import logging
import shutil


def generate_tts_dataset(clips_dir: Path, output_dir: Path):
    """
    Creates a complete TTS training dataset with CSV metadata and audio files.

    This function creates a self-contained dataset directory containing:
    - dataset.csv: CSV file mapping audio filenames to text transcripts
    - Audio files: All .wav clips copied from the clips directory

    The resulting dataset/ directory can be easily exported to platforms like HuggingFace.

    Args:
        clips_dir (Path): Directory containing .wav and .txt files from segmentation
        output_dir (Path): Base output directory where dataset/ folder will be created

    Returns:
        bool: True if dataset was created successfully, False otherwise

    Dataset structure:
        output_dir/
        └── dataset/
            ├── dataset.csv
            ├── 01.wav
            ├── 02.wav
            └── ...
    """
    logger = logging.getLogger(__name__)
    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)

    # Create dataset directory
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(clips_dir.glob("*.wav"))

    if not wav_files:
        logger.warning("No .wav files found in %s", clips_dir)
        return False

    logger.info("Creating TTS dataset with %d clips in %s", len(wav_files), dataset_dir)

    # CSV file path inside dataset directory
    csv_path = dataset_dir / "dataset.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        # Write UTF-8 BOM to ensure proper encoding recognition
        csvfile.write('\ufeff')

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

            # Copy audio file to dataset directory
            dest_wav = dataset_dir / wav_path.name
            try:
                shutil.copy2(wav_path, dest_wav)
                logger.debug("Copied audio file: %s -> %s", wav_path.name, dest_wav)
            except Exception as e:
                logger.error("Failed to copy %s: %s", wav_path.name, e)
                continue

            # Add entry to CSV
            writer.writerow([wav_path.name, text])
            count += 1

    logger.info("Dataset created successfully: %d files in %s", count, dataset_dir)
    logger.info("CSV metadata: %s", csv_path)

    if count == 0:
        logger.warning("No valid pairs were found; dataset is empty.")
        return False

    return True
