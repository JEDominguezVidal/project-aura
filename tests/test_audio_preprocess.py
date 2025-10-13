#!/usr/bin/env python3
"""
Test script for audio preprocessing functionality.

Tests ffmpeg installation and audio conversion to mono 16kHz 16-bit WAV format.
Uses the sample.wav file from assets/ directory.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_preprocess import ensure_wav_for_whisper
from core.utils import setup_logger


def test_audio_preprocessing():
    """Test audio preprocessing functionality."""
    print("🧪 Testing Audio Preprocessing...")
    print()

    # Setup paths
    project_root = Path(__file__).parent.parent
    sample_wav = project_root / "assets" / "sample.wav"
    output_wav = project_root / "tests" / "test_output" / "preprocessed_audio.wav"

    # Ensure output directory exists
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Check if sample file exists
    if not sample_wav.exists():
        print("❌ Sample WAV file not found at assets/sample.wav")
        return False

    print(f"📁 Using sample file: {sample_wav}")

    try:
        # Setup logger
        logger = setup_logger()

        # Test audio preprocessing
        print("🔄 Converting WAV to mono 16kHz WAV...")
        ensure_wav_for_whisper(logger, sample_wav, output_wav)

        # Verify output
        if output_wav.exists() and output_wav.stat().st_size > 0:
            print("✅ Audio preprocessing successful")
            print(f"📄 Output file created: {output_wav}")
            print(f"📏 File size: {output_wav.stat().st_size} bytes")
            return True
        else:
            print("❌ Output file not created or is empty")
            return False

    except Exception as e:
        print(f"❌ Audio preprocessing failed: {e}")
        return False

    finally:
        # Clean up
        if output_wav.exists():
            try:
                output_wav.unlink()
                print("🧹 Cleaned up test output file")
            except:
                pass

        # Clean up test_output directory if empty
        test_output_dir = output_wav.parent
        if test_output_dir.exists() and not any(test_output_dir.iterdir()):
            try:
                test_output_dir.rmdir()
                print("🧹 Cleaned up test_output directory")
            except:
                pass


if __name__ == "__main__":
    success = test_audio_preprocessing()
    sys.exit(0 if success else 1)
