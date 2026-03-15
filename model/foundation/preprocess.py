"""Audio and text preprocessing utilities for Somali ASR."""

import json
import re
from pathlib import Path

import torch
import torchaudio


def normalize_text(text: str) -> str:
    """Normalize Somali text for ASR training.

    - Lowercases
    - Removes punctuation (keeps apostrophes used in Somali)
    - Collapses whitespace
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z ']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Vocabulary ---

# CTC blank token is always index 0
BLANK_TOKEN = "<blank>"


def build_vocab(texts: list[str], vocab_path: str) -> dict[str, int]:
    """Build a character-level vocabulary from training texts.

    Index 0 is reserved for the CTC blank token.
    """
    all_chars = set()
    for text in texts:
        all_chars.update(set(normalize_text(text)))

    # Index 0 = CTC blank
    vocab = {BLANK_TOKEN: 0}
    for idx, char in enumerate(sorted(all_chars), start=1):
        vocab[char] = idx

    vocab_file = Path(vocab_path)
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary saved to {vocab_path} ({len(vocab)} tokens)")
    return vocab


def load_vocab(vocab_path: str) -> dict[str, int]:
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def text_to_indices(text: str, vocab: dict[str, int]) -> list[int]:
    """Convert a normalized text string to a list of vocab indices."""
    return [vocab[ch] for ch in text if ch in vocab]


def indices_to_text(indices: list[int], vocab: dict[str, int]) -> str:
    """Convert vocab indices back to text, skipping blanks."""
    idx_to_char = {v: k for k, v in vocab.items()}
    return "".join(idx_to_char.get(i, "") for i in indices if i != 0)


# --- Audio features ---

_resamplers: dict = {}
_mel_transforms: dict = {}


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
    target_sample_rate: int = 16_000,
) -> torch.Tensor:
    """Extract a log-mel spectrogram from a waveform.

    Args:
        waveform: Raw audio tensor [channels, samples]
        sample_rate: Original sample rate
        n_mels: Number of mel filterbanks
        target_sample_rate: Resample to this rate if needed

    Returns:
        Log-mel spectrogram tensor [n_mels, time]
    """
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sample_rate != target_sample_rate:
        key = (sample_rate, target_sample_rate)
        if key not in _resamplers:
            _resamplers[key] = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = _resamplers[key](waveform)

    mel_key = (target_sample_rate, n_mels)
    if mel_key not in _mel_transforms:
        _mel_transforms[mel_key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160,
            win_length=400,
        )

    mel = _mel_transforms[mel_key](waveform)  # [1, n_mels, time]
    log_mel = torch.log(mel.clamp(min=1e-9))
    return log_mel.squeeze(0)  # [n_mels, time]
