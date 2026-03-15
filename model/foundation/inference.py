"""Inference script for the trained Somali ASR model.

Transcribe Somali audio files to text using your trained model.
Supports greedy decoding or beam search with optional LM rescoring.

Usage:
    python -m model.foundation.inference --audio path/to/audio.wav
    python -m model.foundation.inference --audio path/to/audio.wav --beam-width 10 --lm ./model/output/lm.pt
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from .architecture import build_model
from .decoder import beam_search_decode, greedy_decode
from .language_model import build_lm
from .preprocess import extract_mel_spectrogram


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained ASR model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    model_config = checkpoint["model_config"]

    model = build_model(model_config, vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocab


def load_lm(lm_path: str, device: str = "cpu"):
    """Load a trained language model from a checkpoint file."""
    checkpoint = torch.load(lm_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    lm_config = checkpoint["lm_config"]

    lm = build_lm(lm_config, vocab_size=len(vocab))
    lm.load_state_dict(checkpoint["model_state_dict"])
    lm.to(device)
    lm.eval()

    return lm


def transcribe(
    audio_path: str,
    model,
    vocab: dict[str, int],
    n_mels: int = 80,
    sampling_rate: int = 16_000,
    device: str = "cpu",
    beam_width: int = 1,
    lm=None,
    lm_weight: float = 0.5,
) -> str:
    """Transcribe a single audio file to text.

    Args:
        audio_path: Path to audio file
        model: Trained SomaliASRModel
        vocab: Character-to-index vocabulary
        n_mels: Number of mel filterbanks
        sampling_rate: Target sample rate
        device: Torch device
        beam_width: 1 = greedy, >1 = beam search
        lm: Trained SomaliLanguageModel (optional)
        lm_weight: Weight for LM rescoring (0-1)
    """
    waveform, sr = torchaudio.load(audio_path)
    mel = extract_mel_spectrogram(waveform, sr, n_mels, sampling_rate)
    mel = mel.unsqueeze(0).to(device)
    mel_length = torch.tensor([mel.shape[2]], dtype=torch.long).to(device)

    with torch.no_grad():
        log_probs, output_lengths = model(mel, mel_length)

    if beam_width > 1:
        texts = beam_search_decode(log_probs, output_lengths, vocab, beam_width, lm, lm_weight)
        return texts[0]

    return greedy_decode(log_probs, vocab)[0]


def main():
    parser = argparse.ArgumentParser(description="Transcribe Somali audio to text")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file or directory")
    parser.add_argument("--model", type=str, default="./model/foundation/output/best.pt",
                        help="Path to ASR model checkpoint")
    parser.add_argument("--lm", type=str, default=None,
                        help="Path to language model checkpoint (enables LM rescoring)")
    parser.add_argument("--beam-width", type=int, default=1,
                        help="Beam width (1=greedy, >1=beam search)")
    parser.add_argument("--lm-weight", type=float, default=0.5,
                        help="LM weight for rescoring (0-1)")
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

    print(f"Loading ASR model from {args.model} on {device}...")
    model, vocab = load_model(args.model, device)

    lm = None
    if args.lm:
        print(f"Loading language model from {args.lm}...")
        lm = load_lm(args.lm, device)

    decode_mode = f"beam={args.beam_width}" if args.beam_width > 1 else "greedy"
    if lm:
        decode_mode += f" + LM (weight={args.lm_weight})"
    print(f"Decoding: {decode_mode}")

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
            text = transcribe(
                str(f), model, vocab, device=device,
                beam_width=args.beam_width, lm=lm, lm_weight=args.lm_weight,
            )
            print(f"{f.name}: {text}")
    else:
        text = transcribe(
            str(audio_path), model, vocab, device=device,
            beam_width=args.beam_width, lm=lm, lm_weight=args.lm_weight,
        )
        print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
