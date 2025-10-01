# Whisper + MFA Dataset Builder

## Project Description

**WORK IN PROGRESS**

This project produces a training dataset of short audio clips and corresponding transcriptions from a single long `.wav` recording (for example, a 1-hour monologue).  
It uses OpenAI Whisper (large-v3) for an initial high-quality transcription and Montreal Forced Aligner (MFA) to obtain precise word/phrase timestamps. The pipeline then slices the original audio into per-sentence clips and emits a text file for every clip plus metadata (CSV/JSON).

Goals:
- Produce clips suitable for supervised training of speech models.
- Provide word/phrase timestamps with millisecond granularity (as far as MFA/aligner precision allows).
- Keep the pipeline modular and reproducible: pre-processing, ASR, forced alignment, segmentation.

