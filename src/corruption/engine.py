"""
Corruption engine: loads an MLX model, applies a strategy to selected
weight tensors, and saves a corrupted copy to disk.

The original model cache is never modified.
"""

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from src.config import RESULTS_DIR
from src.corruption.strategies import CorruptionStrategy
from src.models.mlx_loader import copy_config, get_weights, load_model, save_weights


# Tensor name fragments that identify weight matrices worth corrupting.
# Norms, biases, and embeddings are skipped by default.
_WEIGHT_KEYWORDS = ("attn", "mlp", "ffn", "proj", "fc", "gate", "up", "down")


def _is_weight_tensor(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _WEIGHT_KEYWORDS) and "norm" not in n


def corrupt_model(
    model_id: str,
    strategy: CorruptionStrategy,
    tensor_filter: Callable[[str], bool] = _is_weight_tensor,
    output_dir: str | None = None,
) -> Path:
    """
    Load *model_id*, corrupt selected tensors, and save a new model directory.

    Args:
        model_id:      HuggingFace repo ID or local path
        strategy:      CorruptionStrategy to apply
        tensor_filter: Predicate (name: str) -> bool selecting tensors to corrupt
        output_dir:    Where to write the corrupted model (default: RESULTS_DIR/<model_name>_corrupted)

    Returns:
        Path to the saved model directory.
    """
    model, _ = load_model(model_id)
    weights = get_weights(model)

    safe_name = model_id.replace("/", "_").replace(":", "_")
    out = Path(output_dir or f"{RESULTS_DIR}/{safe_name}_corrupted")

    corrupted_count = 0
    corrupted_weights: dict[str, mx.array] = {}

    for name, tensor in weights.items():
        if tensor_filter(name):
            corrupted_weights[name] = strategy.apply(tensor)
            corrupted_count += 1
        else:
            corrupted_weights[name] = tensor

    mx.eval(corrupted_weights)

    save_weights(corrupted_weights, out)
    copy_config(model_id, out)

    print(f"[INFO] Corrupted {corrupted_count}/{len(weights)} tensors.")
    print(f"[INFO] Saved corrupted model → {out}")
    return out
