import torch
from typing import Tuple, Union, Sequence


TensorLike = Union[torch.Tensor, Sequence[int], Sequence[Sequence[int]]]


def _to_tensor(pred_ids: TensorLike, target_ids: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert arbitrary tensor-like prediction/target containers to tensors
    and validate that they share the same shape.
    """
    pred = pred_ids if isinstance(pred_ids, torch.Tensor) else torch.tensor(pred_ids, dtype=torch.long)
    target = target_ids if isinstance(target_ids, torch.Tensor) else torch.tensor(target_ids, dtype=torch.long)

    if pred.shape != target.shape:
        raise ValueError(f"pred_ids and target_ids must have the same shape, "
                         f"got {tuple(pred.shape)} vs {tuple(target.shape)}")

    return pred, target


def compute_token_accuracy(pred_ids: TensorLike, target_ids: TensorLike, pad_id: int) -> float:
    """
    Compute token-level accuracy, ignoring padding tokens.

    This mirrors the token accuracy logic used in the training loop:
    predictions are compared against targets element-wise, and positions
    where the target is PAD are excluded from both the numerator and
    denominator.

    Parameters
    ----------
    pred_ids:
        Predicted token IDs. Can be a flattened tensor of shape
        `(batch * seq_len,)` or a 2D tensor of shape `(batch, seq_len)`.
        Python sequences with equivalent shapes are also accepted.
    target_ids:
        Target token IDs, with the same shape and type constraints as
        `pred_ids`.
    pad_id:
        The integer ID used for padding (e.g. `config.PAD_ID`).

    Returns
    -------
    float
        Token-level accuracy in \\([0, 1]\\). If there are no non-padding
        tokens, returns `0.0`.
    """
    pred, target = _to_tensor(pred_ids, target_ids)

    # Flatten to handle both flattened and (batch, seq_len) inputs uniformly
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    non_pad_mask = target_flat != pad_id
    total_non_pad = int(non_pad_mask.sum().item())
    if total_non_pad == 0:
        return 0.0

    correct = (pred_flat == target_flat) & non_pad_mask
    num_correct = int(correct.sum().item())

    return num_correct / total_non_pad


def compute_exact_match(
    pred_ids: TensorLike,
    target_ids: TensorLike,
    pad_id: int,
    sos_id: int = None,
    eos_id: int = None,
) -> float:
    """
    Compute sequence-level exact match accuracy for a batch.

    A prediction is considered correct if the entire non-special-token sequence
    of token IDs matches the target sequence exactly. PAD, SOS, and EOS tokens
    are removed independently from each sequence before comparison.

    Parameters
    ----------
    pred_ids:
        Predicted token IDs of shape `(batch, seq_len)` or `(seq_len,)`.
        If 1D, it is treated as a batch of size 1. Python sequences with
        equivalent shapes are also accepted.
    target_ids:
        Target token IDs with the same shape and type constraints as
        `pred_ids`.
    pad_id:
        The integer ID used for padding (e.g. `config.PAD_ID`).
    sos_id:
        Optional integer ID for start-of-sequence token. If provided, SOS
        tokens will be stripped before comparison.
    eos_id:
        Optional integer ID for end-of-sequence token. If provided, EOS
        tokens will be stripped before comparison.

    Returns
    -------
    float
        Fraction of sequences in the batch that are exact matches
        (after stripping PAD, SOS, and EOS tokens), in \\([0, 1]\\). If the
        batch is empty, returns `0.0`.
    """
    pred, target = _to_tensor(pred_ids, target_ids)

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if pred.dim() != 2:
        raise ValueError(
            f"compute_exact_match expects inputs of shape (batch, seq_len) "
            f"or (seq_len,), got tensor with shape {tuple(pred.shape)}"
        )

    batch_size = pred.size(0)
    if batch_size == 0:
        return 0.0

    num_exact = 0
    for b in range(batch_size):
        pred_seq = pred[b]
        target_seq = target[b]

        # Strip special tokens (PAD, SOS, EOS) independently for prediction and target
        def _strip_special(seq: torch.Tensor) -> torch.Tensor:
            mask = seq != pad_id
            if sos_id is not None:
                mask = mask & (seq != sos_id)
            if eos_id is not None:
                mask = mask & (seq != eos_id)
            return seq[mask]

        pred_stripped = _strip_special(pred_seq)
        target_stripped = _strip_special(target_seq)

        # Sequences must have the same effective length and all tokens equal
        if pred_stripped.size(0) != target_stripped.size(0):
            continue
        if torch.equal(pred_stripped, target_stripped):
            num_exact += 1

    return num_exact / batch_size


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    This is a standard dynamic programming implementation using a
    two-row DP table for O(min(len(s1), len(s2))) memory.
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Ensure s1 is the shorter string to minimize memory usage
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    previous_row = list(range(len1 + 1))
    current_row = [0] * (len1 + 1)

    for j in range(1, len2 + 1):
        current_row[0] = j
        c2 = s2[j - 1]
        for i in range(1, len1 + 1):
            c1 = s1[i - 1]
            cost = 0 if c1 == c2 else 1
            deletion = previous_row[i] + 1
            insertion = current_row[i - 1] + 1
            substitution = previous_row[i - 1] + cost
            current_row[i] = min(deletion, insertion, substitution)

        previous_row, current_row = current_row, previous_row

    return previous_row[len1]


def compute_edit_distance(
    pred_ids: TensorLike,
    target_ids: TensorLike,
    tokenizer,
    pad_id: int,
    sos_id: int,
    eos_id: int,
) -> Tuple[float, float]:
    """
    Compute average and normalized Levenshtein edit distance between predicted
    and target LaTeX strings.

    Token sequences are first decoded to LaTeX strings using the provided
    tokenizer. Special tokens (PAD, SOS, EOS) are stripped prior to
    comparison. Distances are computed at the character level.

    Parameters
    ----------
    pred_ids:
        Predicted token IDs of shape `(batch, seq_len)` or `(seq_len,)`,
        or an equivalent Python sequence. Values should be integer token
        IDs produced by the model (e.g. via `logits.argmax(-1)`).
    target_ids:
        Target token IDs with the same shape and type constraints as
        `pred_ids`.
    tokenizer:
        Tokenizer instance exposing a `decode(List[int]) -> str` method
        compatible with the training data (e.g. `LaTeXTokenizer`).
    pad_id:
        ID of the padding token.
    sos_id:
        ID of the start-of-sequence token.
    eos_id:
        ID of the end-of-sequence token.

    Returns
    -------
    (float, float)
        A tuple `(average_edit_distance, average_normalized_edit_distance)`:

        - `average_edit_distance`: mean Levenshtein distance over the batch.
        - `average_normalized_edit_distance`: mean over the batch of
          per-sample distances normalized by `max(len(pred), len(target))`
          (in characters). For pairs where both decoded strings are empty,
          the normalized distance is defined as `0.0`.
    """
    pred, target = _to_tensor(pred_ids, target_ids)

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if pred.dim() != 2:
        raise ValueError(
            f"compute_edit_distance expects inputs of shape (batch, seq_len) "
            f"or (seq_len,), got tensor with shape {tuple(pred.shape)}"
        )

    # Work on CPU for safety and to interoperate with Python string logic
    pred_cpu = pred.detach().cpu()
    target_cpu = target.detach().cpu()

    batch_size = pred_cpu.size(0)
    if batch_size == 0:
        return 0.0, 0.0

    total_distance = 0.0
    total_normalized = 0.0

    for b in range(batch_size):
        pred_seq = pred_cpu[b]
        target_seq = target_cpu[b]

        # Manually strip special tokens before decoding, in case the tokenizer
        # implementation changes or does not handle all of them.
        def _strip_special(seq: torch.Tensor) -> torch.Tensor:
            mask = (seq != pad_id) & (seq != sos_id) & (seq != eos_id)
            return seq[mask]

        pred_clean = _strip_special(pred_seq).tolist()
        target_clean = _strip_special(target_seq).tolist()

        pred_str = tokenizer.decode(pred_clean) if hasattr(tokenizer, "decode") else ""
        target_str = tokenizer.decode(target_clean) if hasattr(tokenizer, "decode") else ""

        dist = float(_levenshtein_distance(pred_str, target_str))
        max_len = max(len(pred_str), len(target_str))
        norm = dist / max_len if max_len > 0 else 0.0

        total_distance += dist
        total_normalized += norm

    avg_distance = total_distance / batch_size
    avg_normalized = total_normalized / batch_size

    return avg_distance, avg_normalized


__all__ = [
    "compute_token_accuracy",
    "compute_exact_match",
    "compute_edit_distance",
]


