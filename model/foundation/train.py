"""Training script for the Somali ASR model.

Trains a CNN + Transformer encoder model from scratch on your own labeled
Somali audio data using CTC (Connectionist Temporal Classification) loss.

Usage:
    python -m model.foundation.train
    python -m model.foundation.train --epochs 100 --batch-size 16 --learning-rate 1e-3
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from .architecture import build_model
from .config import DataConfig, ModelConfig, TrainingConfig
from .dataset import create_dataloaders, load_splits
from .decoder import greedy_decode
from .preprocess import (
    build_vocab,
    indices_to_text,
    normalize_text,
)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate."""
    total_words = 0
    total_errors = 0
    for pred, ref in zip(predictions, references):
        ref_words = ref.split()
        pred_words = pred.split()
        total_words += max(len(ref_words), 1)

        # Simple edit distance at word level
        d = _edit_distance(ref_words, pred_words)
        total_errors += d
    return total_errors / max(total_words, 1)


def _edit_distance(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def evaluate(model, loader, ctc_loss, vocab, device):
    """Run evaluation and return average loss and WER."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for mels, mel_lengths, labels, label_lengths in loader:
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            log_probs, output_lengths = model(mels, mel_lengths)
            loss = ctc_loss(log_probs, labels, output_lengths, label_lengths)
            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            preds = greedy_decode(log_probs, vocab)
            all_preds.extend(preds)

            # Decode references
            for i in range(labels.shape[0]):
                ref_indices = labels[i, :label_lengths[i]].tolist()
                all_refs.append(indices_to_text(ref_indices, vocab))

    avg_loss = total_loss / max(num_batches, 1)
    wer = compute_wer(all_preds, all_refs)
    model.train()
    return avg_loss, wer


def train(model_config: ModelConfig, data_config: DataConfig, training_config: TrainingConfig):
    """Run the full training pipeline."""

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("Somali ASR — Training from Scratch")
    print(f"  Device:    {device}")
    print(f"  Data dir:  {data_config.data_dir}")
    print(f"  Output:    {model_config.output_dir}")
    print("=" * 60)

    torch.manual_seed(training_config.seed)

    # --- Step 1: Load data ---
    print("\n[1/4] Loading data...")
    splits = load_splits(data_config)
    print(f"  Train: {len(splits['train'])} samples")
    if "eval" in splits:
        print(f"  Eval:  {len(splits['eval'])} samples")

    # --- Step 2: Build vocabulary ---
    print("\n[2/4] Building vocabulary...")
    train_texts = [normalize_text(e["sentence"]) for e in splits["train"]]
    vocab = build_vocab(train_texts, model_config.vocab_path)
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")

    # --- Step 3: Build model ---
    print("\n[3/4] Building model...")
    model = build_model(model_config, vocab_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # --- Step 4: Train ---
    print("\n[4/4] Training...")
    loaders = create_dataloaders(
        splits, vocab, model_config, data_config, training_config.batch_size,
    )

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    output_dir = Path(model_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_wer = float("inf")
    patience_counter = 0

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (mels, mel_lengths, labels, label_lengths) in enumerate(loaders["train"], 1):
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            log_probs, output_lengths = model(mels, mel_lengths)
            loss = ctc_loss(log_probs, labels, output_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % training_config.log_interval == 0:
                avg = epoch_loss / num_batches
                print(f"  Epoch {epoch} [{batch_idx}/{len(loaders['train'])}] loss={avg:.4f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        log_msg = f"Epoch {epoch}/{training_config.epochs}  train_loss={avg_train_loss:.4f}  time={elapsed:.1f}s"

        # Evaluate
        if "eval" in loaders and epoch % training_config.eval_interval == 0:
            eval_loss, eval_wer = evaluate(model, loaders["eval"], ctc_loss, vocab, device)
            scheduler.step(eval_wer)
            log_msg += f"  eval_loss={eval_loss:.4f}  WER={eval_wer:.4f}"

            if eval_wer < best_wer:
                best_wer = eval_wer
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab": vocab,
                    "model_config": model_config,
                    "wer": best_wer,
                }, output_dir / "best.pt")
                log_msg += " *best*"
            else:
                patience_counter += 1

        print(log_msg)

        # Save periodic checkpoint
        if epoch % training_config.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": vocab,
                "model_config": model_config,
            }, output_dir / f"checkpoint-{epoch}.pt")

        # Early stopping
        if patience_counter >= training_config.patience:
            print(f"\nEarly stopping after {epoch} epochs (no improvement for {training_config.patience} epochs)")
            break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "model_config": model_config,
    }, output_dir / "final.pt")
    print(f"\nTraining complete. Best WER: {best_wer:.4f}")
    print(f"Models saved to {output_dir}/")

    # Test set evaluation
    if "test" in loaders:
        test_loss, test_wer = evaluate(model, loaders["test"], ctc_loss, vocab, device)
        print(f"\nTest set — loss={test_loss:.4f}  WER={test_wer:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Somali ASR model from scratch")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()

    if args.output_dir:
        model_config.output_dir = args.output_dir
    if args.data_dir:
        data_config.data_dir = args.data_dir
    if args.train_manifest:
        data_config.train_manifest = args.train_manifest
    if args.eval_manifest:
        data_config.eval_manifest = args.eval_manifest
    if args.epochs:
        training_config.epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.seed:
        training_config.seed = args.seed

    train(model_config, data_config, training_config)


if __name__ == "__main__":
    main()
