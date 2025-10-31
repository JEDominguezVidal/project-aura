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
import pandas as pd
from datasets import Dataset, Features, Audio, Value, DatasetDict
from huggingface_hub import login


def generate_tts_dataset(clips_dir: Path, output_dir: Path, resume: bool = False) -> bool:
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
    if not resume and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Find max existing clip number if resuming
    max_existing = 0
    if resume:
        existing_wavs = list(dataset_dir.glob("*.wav"))
        clip_nums = []
        for wav in existing_wavs:
            try:
                num = int(wav.stem)
                clip_nums.append(num)
            except ValueError:
                pass
        if clip_nums:
            max_existing = max(clip_nums)

    wav_files = sorted(clips_dir.glob("*.wav"))

    if not wav_files:
        logger.warning("No .wav files found in %s", clips_dir)
        return False

    logger.info("Creating TTS dataset with %d clips in %s", len(wav_files), dataset_dir)
    if resume:
        logger.info("Resume mode: skipping clips <= %d", max_existing)

    # CSV file path inside dataset directory
    csv_path = dataset_dir / "dataset.csv"
    csv_mode = "a" if resume and csv_path.exists() else "w"
    write_bom = not resume

    with open(csv_path, csv_mode, newline="", encoding="utf-8") as csvfile:
        if write_bom:
            # Write UTF-8 BOM to ensure proper encoding recognition
            csvfile.write('\ufeff')
        writer = csv.writer(csvfile)
        if not resume or not csv_path.exists():
            # Write header only if new file or not resuming
            writer.writerow(["filename", "text"])

        count = 0
        for wav_path in wav_files:
            # Extract clip number from filename
            try:
                clip_num = int(wav_path.stem)
            except ValueError:
                logger.warning("Invalid clip filename %s, skipping", wav_path.name)
                continue

            if resume and clip_num <= max_existing:
                logger.debug("Skipping existing clip %s", wav_path.name)
                continue

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

            # Add entry to CSV (quote text to handle commas safely)
            writer.writerow([wav_path.name, f'"{text}"'])
            count += 1

    logger.info("Dataset created successfully: %d files in %s", count, dataset_dir)
    logger.info("CSV metadata: %s", csv_path)

    if count == 0:
        logger.warning("No valid pairs were found; dataset is empty.")
        return False

    return True


def upload_to_hf(repo_name: str, token: str, dataset_dir: Path) -> bool:
    """
    Upload the generated TTS dataset to HuggingFace Hub.

    This function converts the local CSV-based dataset into a HuggingFace Dataset format
    and uploads it to the specified repository. The dataset is split into train/validation
    sets if it contains more than 1000 samples. Audio files are automatically handled
    with proper sampling rate detection.

    Args:
        repo_name (str): HuggingFace repository name in format 'username/repo-name'.
                        Repository will be created if it doesn't exist (requires proper permissions).
        token (str): HuggingFace authentication token with write permissions.
                    Can be obtained from https://huggingface.co/settings/tokens
        dataset_dir (Path): Local directory containing the dataset (must contain dataset.csv and .wav files).
                           This is the path returned by generate_tts_dataset().

    Returns:
        bool: True if upload was successful, False otherwise.

    Raises:
        SystemExit: Called internally on critical errors with appropriate error codes.

    Example:
        >>> success = upload_to_hf(
        ...     repo_name="myusername/spanish-tts-dataset",
        ...     token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        ...     dataset_dir=Path("./output/dataset")
        ... )
        >>> if success:
        ...     print("Dataset uploaded successfully")
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not repo_name or not isinstance(repo_name, str):
        logger.error("Invalid repository name: %s", repo_name)
        return False

    if not token or not isinstance(token, str):
        logger.error("Invalid HuggingFace token provided")
        return False

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        logger.error("Dataset directory %s does not exist or is not a directory", dataset_dir)
        return False

    logger.info("Preparing dataset upload from: %s", dataset_dir)

    # Find CSV file
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV metadata file found in dataset directory %s", dataset_dir)
        logger.error("Please run generate_tts_dataset() first to create the metadata")
        return False

    if len(csv_files) > 1:
        logger.warning("Multiple CSV files found, using: %s", csv_files[0])
        logger.warning("Using first CSV file found: %s", csv_files[0])

    csv_path = csv_files[0]
    logger.info("Loading metadata from: %s", csv_path)

    try:
        # Read CSV with error handling
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d entries from CSV", len(df))
    except Exception as e:
        logger.error("Failed to read CSV file %s: %s", csv_path, e)
        return False

    # Validate required columns
    required_cols = ['filename', 'text']
    if not all(col in df.columns for col in required_cols):
        logger.error("CSV missing required columns. Expected: %s, Found: %s",
                    required_cols, list(df.columns))
        return False

    if len(df) == 0:
        logger.error("CSV file is empty, no data to upload")
        return False

    # Convert filename column to absolute paths for HF Datasets
    logger.info("Converting filenames to absolute paths...")
    df['audio'] = df['filename'].apply(lambda x: str(dataset_dir / x))
    df = df.drop('filename', axis=1)

    logger.info("DataFrame prepared with %d entries", len(df))

    try:
        # Authenticate with HuggingFace
        logger.info("Authenticating with HuggingFace Hub...")
        login(token)
        logger.info("Authentication successful")
    except Exception as e:
        logger.error("Failed to authenticate with HuggingFace: %s", e)
        logger.error("Verify your token has write permissions")
        return False

    try:
        # Define dataset features for proper type handling
        features = Features({
            "audio": Audio(sampling_rate=24000),  # Default 24kHz, can be overridden
            "text": Value("string"),
        })

        # Convert pandas DataFrame to HuggingFace Dataset
        logger.info("Converting to HuggingFace Dataset format...")
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = ds.cast(features)

        # Determine train/validation split
        min_split_size = 1000
        if len(df) > min_split_size:
            logger.info("Dataset large enough (%d > %d), splitting into train/validation sets",
                       len(df), min_split_size)
            ds_split = ds.train_test_split(test_size=0.05, seed=42)
            dataset_dict = DatasetDict({
                "train": ds_split['train'],
                "validation": ds_split['test']
            })
            logger.info("Train set: %d samples, Validation set: %d samples",
                       len(ds_split['train']), len(ds_split['test']))
        else:
            logger.info("Dataset small (%d <= %d), using entire dataset as training set",
                       len(df), min_split_size)
            dataset_dict = DatasetDict({"train": ds})

    except Exception as e:
        logger.error("Failed to create HuggingFace Dataset: %s", e)
        return False

    # Upload to Hub
    logger.info("Starting upload to HuggingFace Hub repository: %s", repo_name)

    try:
        dataset_dict.push_to_hub(repo_name, token=token)
        logger.info("Dataset uploaded successfully to: https://huggingface.co/%s", repo_name)
    except Exception as e:
        logger.error("Failed to upload dataset: %s", e)
        logger.error("Check repository permissions and network connection")
        return False

    logger.info("Upload completed successfully")
    return True
