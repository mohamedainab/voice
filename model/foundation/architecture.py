"""Somali ASR model architecture — CNN + Transformer Encoder + CTC.

  1. CNN layers downsample mel spectrograms and extract local features
  2. Positional encoding adds sequence position information
  3. Transformer encoder layers model long-range temporal context
  4. Linear layer projects to vocabulary size for CTC decoding
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SomaliASRModel(nn.Module):
    """CNN + Transformer Encoder speech recognition model with CTC output."""

    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        cnn_channels: list[int],
        cnn_kernel_sizes: list[int],
        cnn_strides: list[int],
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        transformer_dropout: float,
    ):
        super().__init__()

        # --- CNN front-end ---
        cnn_layers = []
        in_ch = n_mels
        for out_ch, kernel, stride in zip(cnn_channels, cnn_kernel_sizes, cnn_strides):
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # --- Projection to Transformer dimension ---
        self.proj = nn.Linear(in_ch, d_model)

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(d_model, dropout=transformer_dropout)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # --- Output projection ---
        self.fc = nn.Linear(d_model, vocab_size)

    def _make_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create a boolean mask where True = padded position."""
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)  # [batch, max_len]

    def forward(self, x: torch.Tensor, mel_lengths: torch.Tensor):
        """Forward pass.

        Args:
            x: Mel spectrogram [batch, n_mels, time]
            mel_lengths: Original time lengths [batch]

        Returns:
            log_probs: [time, batch, vocab_size] (for CTC loss)
            output_lengths: [batch]
        """
        # CNN: [batch, n_mels, time] -> [batch, cnn_channels, time']
        x = self.cnn(x)
        output_lengths = mel_lengths.clone()
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                output_lengths = (
                    (output_lengths + 2 * layer.padding[0] - layer.kernel_size[0])
                    // layer.stride[0]
                    + 1
                )

        # [batch, channels, time'] -> [batch, time', channels]
        x = x.transpose(1, 2)

        # Project to d_model
        x = self.proj(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder with padding mask
        padding_mask = self._make_padding_mask(output_lengths, x.size(1))
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Project to vocab
        x = self.fc(x)
        log_probs = x.log_softmax(dim=-1)

        # CTC expects [time, batch, vocab_size]
        log_probs = log_probs.transpose(0, 1)

        return log_probs, output_lengths


def build_model(model_config, vocab_size: int) -> SomaliASRModel:
    """Construct a model from config."""
    return SomaliASRModel(
        n_mels=model_config.n_mels,
        vocab_size=vocab_size,
        cnn_channels=model_config.cnn_channels,
        cnn_kernel_sizes=model_config.cnn_kernel_sizes,
        cnn_strides=model_config.cnn_strides,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_encoder_layers=model_config.num_encoder_layers,
        dim_feedforward=model_config.dim_feedforward,
        transformer_dropout=model_config.transformer_dropout,
    )
