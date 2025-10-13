#!/usr/bin/env python3
"""
Test script for audio segmentation functionality.

Tests ffmpeg clip generation from aligned audio segments.
Uses the sample.wav file from assets/ directory.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_preprocess import ensure_wav_for_whisper
from core.asr_whisper import transcribe_whisper
from core.align_mfa import run_mfa_alignment, parse_textgrid_for_sentences
from core.segmenter import export_sentence_clips
from core.utils import setup_logger


def test_segmenter():
    """Test audio segmentation functionality."""
    print("🧪 Testing Audio Segmentation...")

    # Setup paths
    project_root = Path(__file__).parent.parent
    sample_wav = project_root / "assets" / "sample.wav"
    preprocessed_wav = project_root / "tests" / "test_output" / "segmenter_test.wav"
    transcript_txt = project_root / "tests" / "test_output" / "segmenter_transcript.txt"
    mfa_output_dir = project_root / "tests" / "test_output" / "mfa_segmenter"
    clips_dir = project_root / "tests" / "test_output" / "clips_test"

    # Ensure output directories exist
    preprocessed_wav.parent.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Check if sample file exists
    if not sample_wav.exists():
        print("❌ Sample WAV file not found at assets/sample.wav")
        return False

    print(f"📁 Using sample file: {sample_wav}")

    try:
        # Setup logger
        logger = setup_logger()

        # First preprocess the audio
        print("🔄 Preprocessing audio for segmentation...")
        ensure_wav_for_whisper(logger, sample_wav, preprocessed_wav)

        if not preprocessed_wav.exists():
            print("❌ Audio preprocessing failed")
            return False

        # Generate transcript
        print("📝 Generating transcript...")
        transcribe_whisper(logger, str(preprocessed_wav), str(transcript_txt))

        if not transcript_txt.exists() or transcript_txt.read_text().strip() == "":
            print("❌ Transcript generation failed")
            return False

        # Run MFA alignment
        print("🎯 Running MFA alignment...")
        textgrid_path = run_mfa_alignment(
            logger=logger,
            wav_path=preprocessed_wav,
            transcript_path=transcript_txt,
            out_dir=mfa_output_dir,
            mfa_lang="spanish_mfa"
        )

        if not textgrid_path or not textgrid_path.exists():
            print("❌ MFA alignment failed")
            return False

        # Parse sentences with timestamps
        print("📋 Parsing aligned sentences...")
        sentences = parse_textgrid_for_sentences(logger, textgrid_path, transcript_txt)

        if not sentences:
            print("❌ No sentences could be aligned")
            return False

        print(f"📝 Found {len(sentences)} aligned sentences")

        # Test segmentation
        print("✂️ Generating audio clips...")
        generated_clips = export_sentence_clips(logger, preprocessed_wav, sentences, clips_dir)

        # Verify results
        if generated_clips and len(generated_clips) > 0:
            print("✅ Audio segmentation successful")
            print(f"🎵 Generated {len(generated_clips)} audio clips")

            # Check if files actually exist
            wav_files = list(clips_dir.glob("*.wav"))
            txt_files = list(clips_dir.glob("*.txt"))

            print(f"📁 WAV files created: {len(wav_files)}")
            print(f"📄 TXT files created: {len(txt_files)}")

            if len(wav_files) > 0 and len(txt_files) > 0:
                return True
            else:
                print("❌ Clip files not found in output directory")
                return False
        else:
            print("❌ No clips were generated")
            return False

    except Exception as e:
        print(f"❌ Segmentation test failed: {e}")
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

        # Clean up directories
        for dir_path in [mfa_output_dir, clips_dir]:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"🧹 Cleaned up {dir_path.name} directory")
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
    success = test_segmenter()
    sys.exit(0 if success else 1)
