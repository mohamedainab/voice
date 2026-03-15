# Somali ASR Model

Trains a speech-to-text model **from scratch** on your own labeled Somali audio data.

## Architecture

- **CNN + Transformer Encoder** — CNN extracts local features, Transformer models long-range context
- **CTC loss** (Connectionist Temporal Classification) for alignment-free training
- **Mel spectrogram** features (80 filterbanks at 16 kHz)
- **Character-level vocabulary** built automatically from your training text
- **Language model** — character-level Transformer LM trained on Somali text for rescoring
- **Beam search decoder** with LM integration for higher accuracy

## Project Structure

```
model/
└── foundation/
    ├── config.py          # Model, LM, data, and training configs
    ├── preprocess.py      # Text normalization, vocabulary, mel spectrogram extraction
    ├── dataset.py         # Dataset loading from TSV manifests, DataLoaders
    ├── architecture.py    # CNN + Transformer Encoder + CTC model
    ├── language_model.py  # Character-level Transformer language model
    ├── decoder.py         # Beam search CTC decoder with LM rescoring
    ├── train.py           # ASR training loop
    ├── train_lm.py        # Language model training loop
    ├── inference.py       # Transcribe audio (greedy or beam search + LM)
    └── README.md
```

## Data Format

Organize your labeled data with a **TSV manifest file** and audio files:

```
data/
├── train.tsv         # Training manifest
├── eval.tsv          # Evaluation manifest (optional — auto-splits 10% from train)
├── test.tsv          # Test manifest (optional)
└── clips/            # Audio files (WAV, MP3, FLAC, etc.)
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

### Manifest Format (tab-separated, with header)

```
audio_path	sentence
clips/0001.wav	magaalada waa weyn tahay
clips/0002.wav	waxaan rabaa inaan baro af soomaali
clips/0003.wav	subax wanaagsan
```

- `audio_path` — relative path to the audio file from `data_dir` (default: `./data`)
- `sentence` — the Somali transcription for that audio clip

## Setup

```bash
pip install torch torchaudio
```

## Training

### Step 1: Train the ASR Model

```bash
python -m model.foundation.train
```

With custom arguments:

```bash
python -m model.foundation.train \
    --data-dir ./data \
    --train-manifest ./data/train.tsv \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 3e-4
```

### Step 2: Train the Language Model

Uses your Somali text corpus (e.g., `data/text/somali-phrases-corpus.txt`) to learn word patterns:

```bash
python -m model.foundation.train_lm
```

With custom arguments:

```bash
python -m model.foundation.train_lm \
    --text-corpus ./data/text/somali-phrases-corpus.txt \
    --epochs 20 \
    --batch-size 64
```

### CLI Options (ASR)

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `./model/foundation/output` | Checkpoint directory |
| `--data-dir` | `./data` | Root directory for audio files |
| `--train-manifest` | `./data/train.tsv` | Training TSV manifest |
| `--eval-manifest` | `./data/eval.tsv` | Evaluation TSV manifest |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--learning-rate` | `3e-4` | Learning rate |
| `--seed` | `42` | Random seed |

## Inference

### Greedy Decoding (fast)

```bash
python -m model.foundation.inference --audio path/to/audio.wav
```

### Beam Search + Language Model (more accurate)

```bash
python -m model.foundation.inference \
    --audio path/to/audio.wav \
    --beam-width 10 \
    --lm ./model/foundation/output/lm.pt \
    --lm-weight 0.5
```

### Transcribe a Directory

```bash
python -m model.foundation.inference --audio path/to/audio_dir/ --beam-width 10 --lm ./model/foundation/output/lm.pt
```

### Use in Python

```python
from model.foundation.inference import load_model, load_lm, transcribe

model, vocab = load_model("./model/foundation/output/best.pt")
lm = load_lm("./model/foundation/output/lm.pt")  # optional

# Greedy
text = transcribe("audio.wav", model, vocab)

# Beam search + LM
text = transcribe("audio.wav", model, vocab, beam_width=10, lm=lm, lm_weight=0.5)
print(text)
```

## Outputs

Training produces these files in `./model/foundation/output/`:

| File | Description |
|---|---|
| `best.pt` | Best model checkpoint (lowest eval WER) |
| `final.pt` | Model state at end of training |
| `checkpoint-N.pt` | Periodic checkpoints every N epochs |
| `vocab.json` | Character vocabulary |

## Hardware

- **GPU recommended**: NVIDIA GPU or Apple Silicon (MPS). CPU works for small datasets.
- **RAM**: 8+ GB recommended.
