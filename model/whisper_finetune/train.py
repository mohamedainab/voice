"""Fine-tuning script for Whisper on Somali speech data.

Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of
OpenAI's Whisper large-v3 model on your own labeled Somali audio data.

Usage:
    python -m model.whisper_finetune.train
    python -m model.whisper_finetune.train --epochs 10 --batch-size 8
"""

import argparse
import time
from pathlib import Path

import torch
import evaluate as hf_evaluate
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import WhisperConfig, WhisperDataConfig, WhisperTrainingConfig
from .dataset import DataCollator, WhisperSomaliDataset, load_splits


def train(
    whisper_config: WhisperConfig,
    data_config: WhisperDataConfig,
    training_config: WhisperTrainingConfig,
):
    """Run the Whisper fine-tuning pipeline."""

    print("=" * 60)
    print("Somali ASR — Whisper Fine-Tuning (LoRA)")
    print(f"  Base model:  {whisper_config.model_name}")
    print(f"  Language:    {whisper_config.language}")
    print(f"  LoRA:        r={whisper_config.lora_r}, alpha={whisper_config.lora_alpha}")
    print(f"  Data dir:    {data_config.data_dir}")
    print(f"  Output:      {whisper_config.output_dir}")
    print("=" * 60)

    # --- Step 1: Load Whisper components ---
    print("\n[1/5] Loading Whisper model and processors...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_config.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        whisper_config.model_name,
        language=whisper_config.language,
        task=whisper_config.task,
    )
    processor = WhisperProcessor.from_pretrained(
        whisper_config.model_name,
        language=whisper_config.language,
        task=whisper_config.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(whisper_config.model_name)

    # Force Somali language and transcribe task tokens during generation
    model.generation_config.language = whisper_config.language
    model.generation_config.task = whisper_config.task
    model.generation_config.forced_decoder_ids = None

    # --- Step 2: Apply LoRA ---
    if whisper_config.use_lora:
        print("\n[2/5] Applying LoRA adapters...")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=whisper_config.lora_r,
            lora_alpha=whisper_config.lora_alpha,
            lora_dropout=whisper_config.lora_dropout,
            target_modules=whisper_config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("\n[2/5] Full fine-tuning (no LoRA)...")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # --- Step 3: Load data ---
    print("\n[3/5] Loading data...")
    splits = load_splits(data_config)
    print(f"  Train: {len(splits['train'])} samples")
    if "eval" in splits:
        print(f"  Eval:  {len(splits['eval'])} samples")

    train_dataset = WhisperSomaliDataset(
        splits["train"], feature_extractor, tokenizer, data_config.sampling_rate
    )
    eval_dataset = None
    if "eval" in splits:
        eval_dataset = WhisperSomaliDataset(
            splits["eval"], feature_extractor, tokenizer, data_config.sampling_rate
        )

    data_collator = DataCollator(tokenizer)

    # --- Step 4: Set up metrics ---
    print("\n[4/5] Setting up training...")
    wer_metric = hf_evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # --- Step 5: Train ---
    print("\n[5/5] Training...")
    output_dir = Path(whisper_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        fp16=training_config.fp16,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        logging_steps=training_config.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="wer" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        seed=training_config.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    callbacks = []
    if eval_dataset and training_config.patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=training_config.patience)
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
        processing_class=processor,
        callbacks=callbacks,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Model saved to {output_dir / 'final'}")

    # Test set evaluation
    if "test" in splits:
        test_dataset = WhisperSomaliDataset(
            splits["test"], feature_extractor, tokenizer, data_config.sampling_rate
        )
        results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print(f"\nTest set — WER: {results['test_wer']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Somali ASR")
    parser.add_argument("--model-name", type=str, default=None,
                        help="HuggingFace model ID (default: openai/whisper-large-v3)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    whisper_config = WhisperConfig()
    data_config = WhisperDataConfig()
    training_config = WhisperTrainingConfig()

    if args.model_name:
        whisper_config.model_name = args.model_name
    if args.output_dir:
        whisper_config.output_dir = args.output_dir
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
    if args.lora_r:
        whisper_config.lora_r = args.lora_r
    if args.no_lora:
        whisper_config.use_lora = False
    if args.seed:
        training_config.seed = args.seed

    train(whisper_config, data_config, training_config)


if __name__ == "__main__":
    main()
