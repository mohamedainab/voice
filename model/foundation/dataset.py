"""Dataset loading for Somali ASR training.

Loads audio + text pairs from local TSV manifest files.
Expected manifest format (tab-separated, with header):

    audio_path\tsentence
    clips/0001.wav\tmagaalada waa weyn tahay
    clips/0002.wav\twaxaan rabaa inaan baro

Audio paths are resolved relative to data_config.data_dir.
"""

import csv
import random
from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .config import DataConfig, ModelConfig
from .preprocess import (
    extract_mel_spectrogram,
    normalize_text,
    text_to_indices,
)


def load_manifest(manifest_path: str, data_dir: str,
                  min_duration: float = 0.0,
                  max_duration: float = float("inf")) -> list[dict]:
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


class SomaliASRDataset(Dataset):
    """PyTorch Dataset that loads audio and returns mel spectrograms + label indices."""

    def __init__(self, entries: list[dict], vocab: dict[str, int],
                 n_mels: int = 80, sampling_rate: int = 16_000):
        self.entries = entries
        self.vocab = vocab
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        waveform, sr = torchaudio.load(entry["audio_path"])
        mel = extract_mel_spectrogram(waveform, sr, self.n_mels, self.sampling_rate)
        # mel shape: [n_mels, time]

        text = normalize_text(entry["sentence"])
        labels = torch.tensor(text_to_indices(text, self.vocab), dtype=torch.long)

        return mel, labels


def collate_fn(batch):
    """Pad mel spectrograms and labels to equal length within a batch.

    Returns:
        mels:        [batch, n_mels, max_time]  (padded)
        mel_lengths: [batch]                    (original time lengths)
        labels:      [batch, max_label_len]     (padded with 0 = blank)
        label_lengths: [batch]                  (original label lengths)
    """
    mels, labels = zip(*batch)

    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    # Pad mels along time axis
    max_mel_len = mel_lengths.max().item()
    n_mels = mels[0].shape[0]
    padded_mels = torch.zeros(len(mels), n_mels, max_mel_len)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.shape[1]] = m

    # Pad labels with 0 (blank)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_mels, mel_lengths, padded_labels, label_lengths


def load_splits(data_config: DataConfig) -> dict[str, list[dict]]:
    """Load train/eval/test manifest entries.

    Auto-splits eval from train if no eval manifest exists.
    """
    dur_args = (data_config.min_audio_length_sec, data_config.max_audio_length_sec)

    train_rows = load_manifest(data_config.train_manifest, data_config.data_dir, *dur_args)
    if not train_rows:
        raise FileNotFoundError(
            f"Training manifest not found or empty: {data_config.train_manifest}\n"
            f"Create a TSV file with columns: audio_path\\tsentence"
        )

    eval_rows = load_manifest(data_config.eval_manifest, data_config.data_dir, *dur_args)

    # Auto-split if no eval manifest
    if not eval_rows and data_config.eval_split_pct > 0:
        random.seed(42)
        random.shuffle(train_rows)
        split_idx = int(len(train_rows) * (1 - data_config.eval_split_pct))
        eval_rows = train_rows[split_idx:]
        train_rows = train_rows[:split_idx]

    test_rows = load_manifest(data_config.test_manifest, data_config.data_dir, *dur_args)

    splits = {"train": train_rows}
    if eval_rows:
        splits["eval"] = eval_rows
    if test_rows:
        splits["test"] = test_rows
    return splits


def create_dataloaders(
    splits: dict[str, list[dict]],
    vocab: dict[str, int],
    model_config: ModelConfig,
    data_config: DataConfig,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Build DataLoaders for each split."""
    loaders = {}
    for name, rows in splits.items():
        ds = SomaliASRDataset(
            rows, vocab,
            n_mels=model_config.n_mels,
            sampling_rate=data_config.sampling_rate,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=data_config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return loaders
