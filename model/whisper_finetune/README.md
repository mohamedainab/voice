# Whisper Fine-Tuning for Somali ASR

Fine-tunes **OpenAI Whisper large-v3** on your own labeled Somali audio data using **LoRA** (Low-Rank Adaptation) for parameter-efficient training.

## Why Whisper?

Whisper large-v3 is one of the most capable open-source speech models available:

- Pre-trained on **680,000+ hours** of multilingual audio
- Encoder-decoder Transformer architecture (better than CTC for low-resource languages)
- Already has some Somali capability from pre-training
- Fine-tuning adapts it specifically to your data and domain

## Why LoRA?

Instead of updating all 1.5B parameters, LoRA injects small trainable adapters into the attention layers:

- **~1-2% of parameters trained** → vastly less memory and compute
- Works on a single GPU with 16GB+ VRAM
- Retains all of Whisper's pre-trained knowledge
- Adapter weights are small (~50MB vs ~6GB full model)

## Setup

```bash
pip install torch torchaudio transformers peft evaluate jiwer
```

## Data Format

Same TSV manifest format as the foundation model:

```
audio_path	sentence
clips/0001.wav	magaalada waa weyn tahay
clips/0002.wav	waxaan rabaa inaan baro af soomaali
```

## Training

```bash
python -m model.whisper_finetune.train
```

With custom arguments:

```bash
python -m model.whisper_finetune.train \
    --data-dir ./data \
    --train-manifest ./data/train.tsv \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4
```

### Use a different Whisper model

```bash
# Smaller model (faster, less accurate)
python -m model.whisper_finetune.train --model-name openai/whisper-medium

# Turbo variant (faster inference)
python -m model.whisper_finetune.train --model-name openai/whisper-large-v3-turbo
```

### Full fine-tuning (no LoRA — requires more GPU memory)

```bash
python -m model.whisper_finetune.train --no-lora
```

### CLI Options

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `openai/whisper-large-v3` | Base Whisper model |
| `--output-dir` | `./model/whisper_finetune/output` | Output directory |
| `--data-dir` | `./data` | Root directory for audio files |
| `--train-manifest` | `./data/train.tsv` | Training TSV manifest |
| `--eval-manifest` | `./data/eval.tsv` | Evaluation TSV manifest |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `8` | Per-device batch size |
| `--learning-rate` | `1e-4` | Learning rate |
| `--lora-r` | `16` | LoRA rank |
| `--no-lora` | `false` | Disable LoRA |
| `--seed` | `42` | Random seed |

## Inference

```bash
python -m model.whisper_finetune.inference --audio path/to/audio.wav
```

Transcribe a directory:

```bash
python -m model.whisper_finetune.inference --audio path/to/audio_dir/
```

### Use in Python

```python
from model.whisper_finetune.inference import load_model, transcribe

model, processor = load_model("./model/whisper_finetune/output/final")
text = transcribe("audio.wav", model, processor)
print(text)
```

## Project Structure

```
model/whisper_finetune/
├── config.py       # Model, data, and training configs
├── dataset.py      # TSV manifest loading, Whisper preprocessing
├── train.py        # LoRA fine-tuning with HuggingFace Trainer
├── inference.py    # Transcribe audio files
└── README.md
```

## Foundation Model vs Whisper Fine-Tune

| | Foundation (`model/foundation/`) | Whisper (`model/whisper_finetune/`) |
|---|---|---|
| Approach | Trained from scratch | Fine-tuned from Whisper large-v3 |
| Architecture | CNN + Transformer + CTC | Encoder-Decoder Transformer |
| Parameters | ~5M | ~1.5B (LoRA trains ~15M) |
| Pre-training | None | 680K hours multilingual |
| Accuracy | Lower (needs lots of data) | Higher (leverages pre-training) |
| Dependencies | torch, torchaudio | + transformers, peft, evaluate |
| GPU Memory | ~4GB | ~16GB (LoRA), ~32GB (full) |

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended (LoRA), or Apple Silicon with 16GB+ unified memory
- **CPU**: Possible but very slow
- **Disk**: ~12GB for Whisper large-v3 weights download
