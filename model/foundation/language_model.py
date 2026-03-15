"""Character-level language model for Somali text.

A small Transformer-based LM trained on Somali text corpus.
Used during inference to rescore beam search candidates from the ASR model.
"""

import math

import torch
import torch.nn as nn

from .architecture import PositionalEncoding


class SomaliLanguageModel(nn.Module):
    """Transformer-based character-level language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token indices [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        mask = self._causal_mask(x.size(1), x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc(x)

    def score_sequence(self, indices: list[int]) -> float:
        """Score a character sequence (log probability).

        Used by beam search to evaluate candidate transcriptions.
        """
        if len(indices) < 2:
            return 0.0

        self.eval()
        device = self.embedding.weight.device
        x = torch.tensor([indices[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor(indices[1:], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = self(x)  # [1, seq_len, vocab_size]
            log_probs = logits.squeeze(0).log_softmax(dim=-1)
            # Sum log probs of each target given its prefix
            score = sum(log_probs[i, targets[i]].item() for i in range(len(targets)))

        return score


def build_lm(lm_config, vocab_size: int) -> SomaliLanguageModel:
    """Construct a language model from config."""
    return SomaliLanguageModel(
        vocab_size=vocab_size,
        d_model=lm_config.d_model,
        nhead=lm_config.nhead,
        num_layers=lm_config.num_layers,
        dim_feedforward=lm_config.dim_feedforward,
        dropout=lm_config.dropout,
    )
