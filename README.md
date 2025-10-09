# Whisper + MFA Dataset Builder

## Project Description

This project produces a training dataset of short audio clips and corresponding transcriptions from a single long `.wav`/`.mp3` recording (for example, a 30-min monologue).  
It uses OpenAI Whisper (large-v3) for an initial high-quality transcription and Montreal Forced Aligner (MFA) to obtain precise word/phrase timestamps. The pipeline then slices the original audio into per-sentence clips and emits a text file for every clip.

Goals:
- Produce clips suitable for supervised training of speech models.
- Provide word/phrase timestamps with millisecond granularity (as far as MFA/aligner precision allows).
- Keep the pipeline modular and reproducible: pre-processing, ASR, forced alignment, segmentation.

This repository is intended for researchers and engineers familiar with command-line tools and Python environments. It assumes you are comfortable installing system tools (FFmpeg, MFA) and, optionally, using a GPU for Whisper inference.


## Project Structure

### File Structure:
```
.
├── README.md
├── requirements.txt
├── main.py                     # CLI entrypoint that orchestrates the pipeline
├── audio_preprocess.py         # audio conversion/preprocessing (ffmpeg wrapper)
├── asr_whisper.py              # Whisper (large-v3) wrapper for initial transcription
├── align_mfa.py                # prepare corpus and run Montreal Forced Aligner (MFA); parse TextGrid
├── segmenter.py                # cut audio into clips per sentence and write .txt files
├── utils.py                    # small utilities (logging, path checks, time formatting)
└── assets/			 # Project assets
    └── sample.mp3              # Sample audio for testing
```

## Installation and Usage (CLI)

### Prerequisites (system)
1. **Python** — Python 3.8+ recommended (3.10/3.11 preferable).  
2. **FFmpeg** — required; used to convert and slice audio. Ensure `ffmpeg` is in your `PATH`.  
   - On macOS (Homebrew): `brew install ffmpeg`  
   - On Ubuntu/Debian: `sudo apt install ffmpeg`  
3. **Montreal Forced Aligner (MFA)** — required for forced alignment. Recommended installation via conda:
   ```bash
   conda install -c conda-forge montreal-forced-aligner
   ```
   After installation, make sure the mfa CLI is available in your shell. You will also need to download the appropriate acoustic model / dictionary for your language (see MFA docs). MFA stores models locally — use the MFA model subcommands or the MFA documentation to fetch Spanish/English models.
4. **GPU (recommended but optional)** — Whisper large-v3 is large. A GPU with ≥24GB VRAM is recommended for reasonable inference time. CPU inference is possible but can be very slow.

### How to install dependencies (Linux):
This section describes how to prepare a working environment for the project and how to run the pipeline from the command line. The instructions assume you will use Conda to install Montreal Forced Aligner (MFA) and system binaries (FFmpeg) and then install Python packages inside the same Conda environment.
If you already have Conda (Anaconda, Miniconda or Miniforge) installed, skip the Miniconda install step and start at Create and activate a Conda environment.

1. Install Miniconda (only if you don't already have conda)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
source ~/.bashrc
```

2. Create and activate a conda environment:
```bash
conda create -n aura python=3.9 -y
conda activate aura
```

3. Add conda-forge and install MFA + ffmpeg:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -y montreal-forced-aligner ffmpeg
```

4. Install PyTorch (choose CPU or the correct CUDA wheel for your GPU):
CPU example:
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
GPU example (for CUDA 12.4. Change accordingly):
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

5. Install project dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

6. Download MFA models (for Spanish. Change accordingly):
```bash
mfa model download acoustic spanish_mfa
mfa model download dictionary spanish_mfa
mfa model download g2p spanish_spain_mfa
```

7. Install NLTK data (sentence tokenizer):
```bash
python -m nltk.downloader punkt
```

### Instalation Verification:
Run these simple checks with the Conda environment activated:
```bash
# Check ffmpeg
ffmpeg -version

# Check mfa
mfa --version

# Check Python packages
python -c "import whisper, textgrid, soundfile; print('python deps ok')"

# Check torch (and CUDA availability)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```


### Usage Examples:
Run the pipeline with the main script:
```bash
python main.py --input /path/to/input.wav --outdir ./dataset_out --lang es --whisper_model large-v3 --mfa_lang spanish_mfa
```

#### Notes and recommendations:
- Transcription accuracy matters for alignment. If the Whisper transcript contains many errors, forced alignment will struggle. Inspect transcript.txt and manually correct obvious errors if you require perfect segmentation.

- Forced aligners provide the best sub-100ms accuracy when transcript and audio match exactly and audio quality is good. Expect typical practical alignment granularity in the 5–50 ms window depending on audio quality and aligner.

- Clip duration: the pipeline is configured to export sentence-level clips. If you need fixed durations (e.g. 4–20 s), modify segmenter.py to group or split sentences accordingly and to add/trim overlaps.

- Silence/overlap: the pipeline adds configurable pre/post roll around sentences to avoid clipping words at boundaries — tune pre_roll and post_roll in segmenter.py.

- Large models & resources: large-v3 is very accurate but resource heavy. If you lack the hardware, use a smaller Whisper model.
