"""
Core functionality for the Whisper+MFA dataset building pipeline.

This package contains the main processing modules:
- config: Centralized configuration parameters
- audio_preprocess: Audio format conversion for Whisper compatibility
- asr_whisper: OpenAI Whisper transcription wrapper
- align_mfa: Montreal Forced Aligner integration
- segmenter: Audio segmentation and clip generation
"""

__version__ = "0.0.1"
