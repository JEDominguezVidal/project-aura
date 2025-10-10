#!/usr/bin/env python3
"""
Test script for Automatic Speech Recognition (ASR) functionality.

Tests Whisper installation and transcription generation.
Uses the sample.wav file from assets/ directory.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_preprocess import ensure_wav_for_whisper
from asr_whisper import transcribe_whisper
from utils import setup_logger


def test_asr():
    """Test ASR functionality."""
    print("🧪 Testing Automatic Speech Recognition...")

    # Setup paths
    project_root = Path(__file__).parent.parent
    sample_wav = project_root / "assets" / "sample.wav"
    preprocessed_wav = project_root / "tests" / "test_output" / "asr_test.wav"
    output_txt = project_root / "tests" / "test_output" / "asr_transcript.txt"

    # Ensure output directory exists
    preprocessed_wav.parent.mkdir(parents=True, exist_ok=True)

    # Check if sample file exists
    if not sample_wav.exists():
        print("❌ Sample WAV file not found at assets/sample.wav")
        return False

    print(f"📁 Using sample file: {sample_wav}")

    try:
        # Setup logger
        logger = setup_logger()

        # First preprocess the audio
        print("🔄 Preprocessing audio for Whisper...")
        ensure_wav_for_whisper(logger, sample_wav, preprocessed_wav)

        if not preprocessed_wav.exists():
            print("❌ Audio preprocessing failed")
            return False

        # Test ASR transcription
        print("🎙️ Running Whisper transcription...")
        result = transcribe_whisper(logger, str(preprocessed_wav), str(output_txt))

        # Verify results
        if output_txt.exists():
            transcript = output_txt.read_text(encoding='utf-8').strip()
            if transcript and len(transcript) > 0:
                print("✅ ASR transcription successful")
                print(f"📝 Transcript length: {len(transcript)} characters")
                print(f"📄 First 100 characters: {transcript[:100]}...")
                print(f"🎯 Detected language: {result.get('language', 'unknown')}")
                return True
            else:
                print("❌ Transcript file is empty")
                return False
        else:
            print("❌ Transcript file not created")
            return False

    except Exception as e:
        print(f"❌ ASR test failed: {e}")
        return False

    finally:
        # Clean up
        for file in [preprocessed_wav, output_txt]:
            if file.exists():
                try:
                    file.unlink()
                    print(f"🧹 Cleaned up {file.name}")
                except:
                    pass

        # Clean up test_output directory if empty
        test_output_dir = preprocessed_wav.parent
        if test_output_dir.exists() and not any(test_output_dir.iterdir()):
            try:
                test_output_dir.rmdir()
                print("🧹 Cleaned up test_output directory")
            except:
                pass


if __name__ == "__main__":
    success = test_asr()
    sys.exit(0 if success else 1)
