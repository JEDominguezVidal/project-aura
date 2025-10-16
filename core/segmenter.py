"""
Audio segmentation functions for creating training clips.

This module handles the final step of the audio processing pipeline: segmenting
long-form audio into sentence-level clips based on precise timestamps from
forced alignment. It uses ffmpeg to extract audio segments and saves corresponding
transcriptions to text files for training datasets.

The module assumes that ffmpeg is installed and available in the system PATH.

Main functions:
- export_sentence_clips: Extract audio clips per sentence with transcriptions

Notes:
- Requires ffmpeg for audio cutting operations
- Clips include configurable pre/post roll padding to avoid cutting words
- Only sentences above minimum duration threshold are exported
- Output format: 01.wav/01.txt, 02.wav/02.txt, etc.
"""
from pathlib import Path
import subprocess
import logging
from core.config import MIN_SENTENCE_DURATION, PRE_ROLL_SECONDS, POST_ROLL_SECONDS


def export_sentence_clips(source_wav: Path, sentences: list, out_dir: Path, min_dur: float = MIN_SENTENCE_DURATION, pre_roll: float = PRE_ROLL_SECONDS, post_roll: float = POST_ROLL_SECONDS, outfreq: int = 16000, starting_number: int = 1) -> list:
    """
    Export audio clips for each sentence using ffmpeg.

    For each sentence with sufficient duration, extracts a clip from the source audio
    with optional pre/post roll padding, and saves both the audio clip and transcription.
    Audio is resampled to the specified output frequency if different from input.

    Args:
        source_wav: Path to the source WAV file
        sentences: List of sentence dictionaries with 'sentence', 'start', 'end' keys
        out_dir: Directory to save clips and transcriptions
        min_dur: Minimum sentence duration in seconds to export
        pre_roll: Seconds to include before sentence start
        post_roll: Seconds to include after sentence end
        outfreq: Output sample rate in Hz for clips

    Returns:
        List of paths to generated clip files
    """
    logger = logging.getLogger(__name__)
    generated_clips = []

    for i, sentence_data in enumerate(sentences):
        sentence = sentence_data['sentence']
        start = sentence_data['start']
        end = sentence_data['end']
        duration = end - start

        if duration < min_dur:
            logger.debug("Skipping sentence too short (%.2fs): %s", duration, sentence[:50])
            continue

        # Calculate cut times with padding
        start_cut = max(0, start - pre_roll)
        end_cut = end + post_roll
        duration = end_cut - start_cut

        # Generate filename
        clip_number = i + starting_number
        clip_name = f"{clip_number:02d}"
        wav_path = out_dir / f"{clip_name}.wav"
        txt_path = out_dir / f"{clip_name}.txt"

        # ffmpeg command to extract clip
        if outfreq == 16000:
            # Use copy codec for speed when no resampling needed
            cmd = [
                "ffmpeg", "-y",
                "-i", str(source_wav),
                "-ss", f"{start_cut:.3f}",
                "-t", f"{duration:.3f}",
                "-c", "copy",  # Copy without re-encoding for speed
                str(wav_path)
            ]
        else:
            # Resample to specified frequency
            cmd = [
                "ffmpeg", "-y",
                "-i", str(source_wav),
                "-ss", f"{start_cut:.3f}",
                "-t", f"{duration:.3f}",
                "-ar", str(outfreq),  # Set output sample rate
                "-ac", "1",  # Ensure mono output
                "-c:a", "pcm_s16le",  # Use PCM 16-bit encoding
                str(wav_path)
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug("Exported clip: %s (%.2f-%.2f)", wav_path, start_cut, end_cut)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to export clip %s: %s", wav_path, e)
            continue

        # Save transcription
        try:
            txt_path.write_text(sentence, encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save transcription %s: %s", txt_path, e)
            continue

        generated_clips.append(str(wav_path))

    logger.info("Generated %d clips in %s", len(generated_clips), out_dir)
    return generated_clips
