"""Beam search CTC decoder with optional language model rescoring.

Replaces greedy decoding with beam search to explore multiple candidate
transcriptions and pick the one with the best combined acoustic + LM score.
"""

import torch

from .preprocess import indices_to_text


def greedy_decode(log_probs: torch.Tensor, vocab: dict[str, int]) -> list[str]:
    """Greedy CTC decoding: pick best token per timestep, collapse repeats, drop blanks."""
    pred_ids = log_probs.argmax(dim=-1)  # [time, batch]
    pred_ids = pred_ids.transpose(0, 1)  # [batch, time]

    texts = []
    for seq in pred_ids:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != prev and idx != 0:
                chars.append(idx)
            prev = idx
        texts.append(indices_to_text(chars, vocab))
    return texts


def beam_search_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    vocab: dict[str, int],
    beam_width: int = 10,
    lm: object = None,
    lm_weight: float = 0.5,
) -> list[str]:
    """Beam search CTC decoding with optional LM rescoring.

    Args:
        log_probs: [time, batch, vocab_size] from ASR model
        output_lengths: [batch] valid timestep counts
        vocab: Character vocabulary
        beam_width: Number of beams to keep
        lm: Trained SomaliLanguageModel (or None for pure acoustic)
        lm_weight: Weight for LM score (0 = no LM, 1 = LM only)

    Returns:
        List of decoded strings, one per batch item
    """
    batch_size = log_probs.size(1)
    results = []

    for b in range(batch_size):
        T = output_lengths[b].item()
        probs = log_probs[:T, b, :]  # [T, vocab_size]
        text = _beam_search_single(probs, vocab, beam_width, lm, lm_weight)
        results.append(text)

    return results


def _beam_search_single(
    log_probs: torch.Tensor,
    vocab: dict[str, int],
    beam_width: int,
    lm: object,
    lm_weight: float,
) -> str:
    """Beam search for a single utterance."""
    # Each beam: (prefix_indices, last_char, acoustic_score)
    # prefix_indices: list of non-blank, non-repeat character indices
    beams = [
        ([], -1, 0.0),  # (prefix, last_token, score)
    ]

    T, V = log_probs.shape

    for t in range(T):
        new_beams = {}

        for prefix, last_tok, score in beams:
            for c in range(V):
                log_p = log_probs[t, c].item()
                new_score = score + log_p

                if c == 0:
                    # Blank: keep prefix unchanged
                    key = (tuple(prefix), -1)  # reset last_tok so next char isn't treated as repeat
                    if key not in new_beams or new_beams[key][2] < new_score:
                        new_beams[key] = (prefix, -1, new_score)
                elif c == last_tok:
                    # Repeat of last char: collapse (keep prefix unchanged)
                    key = (tuple(prefix), c)
                    if key not in new_beams or new_beams[key][2] < new_score:
                        new_beams[key] = (prefix, c, new_score)
                else:
                    # New character: extend prefix
                    new_prefix = prefix + [c]
                    key = (tuple(new_prefix), c)
                    if key not in new_beams or new_beams[key][2] < new_score:
                        new_beams[key] = (new_prefix, c, new_score)

        # Keep top beams
        beams = sorted(new_beams.values(), key=lambda x: x[2], reverse=True)[:beam_width]

    # Rescore with LM if available
    if lm is not None and lm_weight > 0:
        scored_beams = []
        for prefix, _, acoustic_score in beams:
            if len(prefix) > 1:
                lm_score = lm.score_sequence(prefix)
            else:
                lm_score = 0.0
            combined = (1 - lm_weight) * acoustic_score + lm_weight * lm_score
            scored_beams.append((prefix, combined))
        scored_beams.sort(key=lambda x: x[1], reverse=True)
        best_prefix = scored_beams[0][0]
    else:
        best_prefix = beams[0][0] if beams else []

    return indices_to_text(best_prefix, vocab)
