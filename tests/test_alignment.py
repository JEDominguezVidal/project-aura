#!/usr/bin/env python3
"""
Test script for MFA alignment functionality.

Tests Montreal Forced Aligner installation and forced alignment capability.
Uses the sample.wav file from assets/ directory.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_preprocess import ensure_wav_for_whisper
from core.asr_whisper import transcribe_whisper
from core.align_mfa import run_mfa_alignment
from core.utils import setup_logger


def test_alignment():
    """Test MFA alignment functionality."""
    print("🧪 Testing MFA Alignment...")

    # Setup paths
    project_root = Path(__file__).parent.parent
    sample_wav = project_root / "assets" / "sample.wav"
    preprocessed_wav = project_root / "tests" / "test_output" / "alignment_test.wav"
    transcript_txt = project_root / "tests" / "test_output" / "alignment_transcript.txt"
    mfa_output_dir = project_root / "tests" / "test_output" / "mfa_test"

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
        print("🔄 Preprocessing audio for MFA...")
        ensure_wav_for_whisper(logger, sample_wav, preprocessed_wav)

        if not preprocessed_wav.exists():
            print("❌ Audio preprocessing failed")
            return False

        # Generate transcript
        print("📝 Generating transcript for alignment...")
        transcribe_whisper(logger, str(preprocessed_wav), str(transcript_txt))

        if not transcript_txt.exists() or transcript_txt.read_text().strip() == "":
            print("❌ Transcript generation failed")
            return False

        # Test MFA alignment
        print("🎯 Running MFA forced alignment...")
        textgrid_path = run_mfa_alignment(
            logger=logger,
            wav_path=preprocessed_wav,
            transcript_path=transcript_txt,
            out_dir=mfa_output_dir,
            mfa_lang="spanish_mfa"
        )

        # Verify results
        if textgrid_path and textgrid_path.exists():
            print("✅ MFA alignment successful")
            print(f"📄 TextGrid file created: {textgrid_path}")
            print(f"📏 File size: {textgrid_path.stat().st_size} bytes")
            return True
        else:
            print("❌ TextGrid file not created")
            return False

    except Exception as e:
        print(f"❌ MFA alignment test failed: {e}")
        return False

    finally:
        # Clean up
        import shutil
        for file in [preprocessed_wav, transcript_txt]:
            if file.exists():
                try:
                    file.unlink()
                    print(f"🧹 Cleaned up {file.name}")
                except:
                    pass

        # Clean up MFA output directory
        if mfa_output_dir.exists():
            try:
                shutil.rmtree(mfa_output_dir)
                print("🧹 Cleaned up MFA output directory")
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
    success = test_alignment()
    sys.exit(0 if success else 1)
