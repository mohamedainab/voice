"""Inference script for the fine-tuned Whisper Somali ASR model.

Transcribe Somali audio using your fine-tuned Whisper model.

Usage:
    python -m model.whisper_finetune.inference --audio path/to/audio.wav
    python -m model.whisper_finetune.inference --audio path/to/audio_dir/
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import PeftModel


def load_model(
    model_path: str,
    base_model_name: str = "openai/whisper-large-v3",
    device: str = "cpu",
    is_lora: bool = True,
):
    """Load a fine-tuned Whisper model.

    Args:
        model_path: Path to the saved model directory
        base_model_name: HuggingFace model ID of the base model
        device: Torch device
        is_lora: Whether the checkpoint is a LoRA adapter

    Returns:
        (model, processor)
    """
    processor = WhisperProcessor.from_pretrained(model_path)

    if is_lora:
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return model, processor


def transcribe(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str = "cpu",
    language: str = "so",
    task: str = "transcribe",
) -> str:
    """Transcribe a single audio file.

    Args:
        audio_path: Path to audio file
        model: Fine-tuned Whisper model
        processor: Whisper processor
        device: Torch device
        language: Language code
        task: 'transcribe' or 'translate'

    Returns:
        Transcribed text
    """
    waveform, sr = torchaudio.load(audio_path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16_000:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)

    input_features = processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16_000,
        return_tensors="pt",
    ).input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task=task
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe Somali audio with fine-tuned Whisper")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file or directory")
    parser.add_argument("--model", type=str, default="./model/whisper_finetune/output/final",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3",
                        help="Base model ID (needed for LoRA)")
    parser.add_argument("--no-lora", action="store_true",
                        help="Model was fully fine-tuned (not LoRA)")
    parser.add_argument("--language", type=str, default="so")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model from {args.model} on {device}...")
    model, processor = load_model(
        args.model,
        base_model_name=args.base_model,
        device=device,
        is_lora=not args.no_lora,
    )

    audio_path = Path(args.audio)

    if audio_path.is_dir():
        extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = sorted(
            f for f in audio_path.iterdir() if f.suffix.lower() in extensions
        )
        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return

        print(f"Found {len(audio_files)} audio files\n")
        for f in audio_files:
            text = transcribe(str(f), model, processor, device=device, language=args.language)
            print(f"{f.name}: {text}")
    else:
        text = transcribe(str(audio_path), model, processor, device=device, language=args.language)
        print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
