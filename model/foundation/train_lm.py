"""Training script for the Somali character-level language model.

Trains on your Somali text corpus so the LM can rescore ASR predictions.

Usage:
    python -m model.foundation.train_lm
    python -m model.foundation.train_lm --text-corpus ./data/text/somali-phrases-corpus.txt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import LMConfig
from .language_model import build_lm
from .preprocess import build_vocab, load_vocab, normalize_text, text_to_indices


class TextDataset:
    """Sliding-window dataset over a character sequence."""

    def __init__(self, indices: list[int], seq_length: int):
        self.indices = indices
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.indices) - self.seq_length)

    def __getitem__(self, idx):
        x = self.indices[idx : idx + self.seq_length]
        y = self.indices[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train_lm(lm_config: LMConfig):
    """Train the character-level language model."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("Somali Language Model Training")
    print(f"  Device: {device}")
    print(f"  Corpus: {lm_config.text_corpus}")
    print("=" * 60)

    torch.manual_seed(lm_config.seed)

    # Load text corpus
    corpus_path = Path(lm_config.text_corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Text corpus not found: {corpus_path}")

    raw_text = corpus_path.read_text(encoding="utf-8")
    text = normalize_text(raw_text)
    print(f"  Corpus: {len(text):,} characters")

    # Build or load vocab (reuse ASR vocab if it exists)
    vocab_path = Path(lm_config.output_path).parent / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab(str(vocab_path))
        print(f"  Reusing ASR vocab ({len(vocab)} tokens)")
    else:
        vocab = build_vocab([text], str(vocab_path))

    vocab_size = len(vocab)

    # Convert text to indices
    indices = text_to_indices(text, vocab)
    print(f"  Encoded: {len(indices):,} tokens")

    # Split train/eval (95/5)
    split_idx = int(len(indices) * 0.95)
    train_ds = TextDataset(indices[:split_idx], lm_config.seq_length)
    eval_ds = TextDataset(indices[split_idx:], lm_config.seq_length)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=lm_config.batch_size, shuffle=True, drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=lm_config.batch_size, shuffle=False, drop_last=True,
    )

    # Build model
    model = build_lm(lm_config, vocab_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lm_config.learning_rate, weight_decay=lm_config.weight_decay,
    )

    output_path = Path(lm_config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")

    for epoch in range(1, lm_config.epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train = total_loss / max(num_batches, 1)

        # Evaluate
        model.eval()
        eval_loss = 0
        eval_batches = 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                eval_loss += loss.item()
                eval_batches += 1

        avg_eval = eval_loss / max(eval_batches, 1)
        elapsed = time.time() - start

        msg = f"Epoch {epoch}/{lm_config.epochs}  train_loss={avg_train:.4f}  eval_loss={avg_eval:.4f}  time={elapsed:.1f}s"

        if avg_eval < best_eval_loss:
            best_eval_loss = avg_eval
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "lm_config": lm_config,
            }, output_path)
            msg += " *saved*"

        print(msg)

    print(f"\nLM training complete. Best eval loss: {best_eval_loss:.4f}")
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Somali language model")
    parser.add_argument("--text-corpus", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    lm_config = LMConfig()
    if args.text_corpus:
        lm_config.text_corpus = args.text_corpus
    if args.epochs:
        lm_config.epochs = args.epochs
    if args.batch_size:
        lm_config.batch_size = args.batch_size
    if args.learning_rate:
        lm_config.learning_rate = args.learning_rate
    if args.output_path:
        lm_config.output_path = args.output_path

    train_lm(lm_config)


if __name__ == "__main__":
    main()
