"""
Configuration parameters for the Whisper+MFA dataset building pipeline.

This module centralizes all configurable parameters used throughout the project,
making it easier to tune the pipeline behavior without modifying multiple files.
"""

# =============================================================================
# CLI DEFAULTS
# =============================================================================

# Default language for transcription
DEFAULT_LANGUAGE = "es"

# Default Whisper model to use
DEFAULT_WHISPER_MODEL = "medium"

# Default MFA language model
DEFAULT_MFA_LANG = "spanish_mfa"

# Default output sample rate in Hz
DEFAULT_OUTPUT_FREQ = 16000

# Default output directory
DEFAULT_OUTPUT_DIR = "./output"

# =============================================================================
# MFA ALIGNMENT PARAMETERS
# =============================================================================

# Beam width for MFA alignment (higher = more accurate but slower)
MFA_BEAM = 100

# Retry beam width for MFA alignment (higher = more accurate but slower)
MFA_RETRY_BEAM = 400

# Minimum ratio of words that must be aligned for a sentence to be considered valid
MIN_WORD_ALIGNMENT_RATIO = 0.2

# =============================================================================
# AUDIO SEGMENTATION PARAMETERS
# =============================================================================

# Minimum sentence duration in seconds to export as a clip
MIN_SENTENCE_DURATION = 0.8

# Seconds to include before sentence start (padding)
PRE_ROLL_SECONDS = 0.2

# Seconds to include after sentence end (padding)
POST_ROLL_SECONDS = 0.2

# =============================================================================
# AUDIO PREPROCESSING PARAMETERS
# =============================================================================

# Target sample rate for Whisper preprocessing (always 16kHz)
WHISPER_SAMPLE_RATE = 16000

# Number of channels for Whisper preprocessing (always mono)
WHISPER_CHANNELS = 1

# Audio encoding for Whisper preprocessing (16-bit PCM)
WHISPER_SAMPLE_FORMAT = "s16"
