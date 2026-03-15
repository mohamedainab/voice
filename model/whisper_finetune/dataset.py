"""Dataset loading for Whisper fine-tuning.

Loads audio + text pairs from the same TSV manifest format used by the
foundation model, then preprocesses them for Whisper (log-mel spectrograms
via WhisperFeatureExtractor, tokenized text via WhisperTokenizer).
"""

import csv
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from transformers import WhisperFeatureExtractor, WhisperTokenizer

from .config import WhisperDataConfig


def load_manifest(
    manifest_path: str,
    data_dir: str,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
) -> list[dict]:
    """Read a TSV manifest and return a list of {audio_path, sentence} dicts."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        return []

    rows = []
    root = Path(data_dir)
    with open(manifest, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_file = root / row["audio_path"]
            if not audio_file.exists():
                print(f"  WARNING: skipping missing file: {audio_file}")
                continue
            info = torchaudio.info(str(audio_file))
            duration = info.num_frames / info.sample_rate
            if duration < min_duration or duration > max_duration:
                continue
            rows.append({
                "audio_path": str(audio_file),
                "sentence": row["sentence"],
            })
    return rows


class WhisperSomaliDataset(Dataset):
    """Dataset that preprocesses audio and text for Whisper fine-tuning."""

    def __init__(
        self,
        entries: list[dict],
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        sampling_rate: int = 16_000,
    ):
        self.entries = entries
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load and resample audio
        waveform, sr = torchaudio.load(entry["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)

        # Extract features (log-mel spectrogram via Whisper's preprocessor)
        input_features = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features.squeeze(0)  # [n_mels, time]

        # Tokenize text
        labels = self.tokenizer(
            entry["sentence"],
            return_tensors="pt",
        ).input_ids.squeeze(0)  # [seq_len]

        return {
            "input_features": input_features,
            "labels": labels,
        }


class DataCollator:
    """Collate batch: pad labels, stack features."""

    def __init__(self, tokenizer: WhisperTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        input_features = torch.stack([b["input_features"] for b in batch])

        # Pad labels to max length in batch, using -100 to ignore in loss
        label_lengths = [b["labels"].shape[0] for b in batch]
        max_label_len = max(label_lengths)

        padded_labels = torch.full(
            (len(batch), max_label_len), -100, dtype=torch.long
        )
        for i, b in enumerate(batch):
            padded_labels[i, : label_lengths[i]] = b["labels"]

        return {
            "input_features": input_features,
            "labels": padded_labels,
        }


def load_splits(data_config: WhisperDataConfig) -> dict[str, list[dict]]:
    """Load train/eval/test manifest entries."""
    dur_args = (data_config.min_audio_length_sec, data_config.max_audio_length_sec)

    train_rows = load_manifest(
        data_config.train_manifest, data_config.data_dir, *dur_args
    )
    if not train_rows:
        raise FileNotFoundError(
            f"Training manifest not found or empty: {data_config.train_manifest}\n"
            f"Create a TSV file with columns: audio_path\tsentence"
        )

    eval_rows = load_manifest(
        data_config.eval_manifest, data_config.data_dir, *dur_args
    )

    # Auto-split if no eval manifest
    if not eval_rows and data_config.eval_split_pct > 0:
        random.seed(42)
        random.shuffle(train_rows)
        split_idx = int(len(train_rows) * (1 - data_config.eval_split_pct))
        eval_rows = train_rows[split_idx:]
        train_rows = train_rows[:split_idx]

    test_rows = load_manifest(
        data_config.test_manifest, data_config.data_dir, *dur_args
    )

    splits = {"train": train_rows}
    if eval_rows:
        splits["eval"] = eval_rows
    if test_rows:
        splits["test"] = test_rows
    return splits
