"""
Load MLX models and expose weights as mx.array dicts for direct manipulation.
"""

import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load as mlx_load
from mlx_lm.utils import get_model_path


def load_model(model_id: str):
    """
    Load an MLX model from a HuggingFace repo ID or local path.

    Returns:
        model:     MLX model object
        tokenizer: associated tokenizer
    """
    print(f"[INFO] Loading model: {model_id}")
    model, tokenizer = mlx_load(model_id)
    mx.eval(model.parameters())
    print(f"[INFO] Model loaded.")
    return model, tokenizer


def get_weights(model) -> dict[str, mx.array]:
    """Return a flat dict of all weight tensors: name → mx.array."""
    return dict(tree_flatten(model.parameters()))


def set_weights(model, weights: dict[str, mx.array]) -> None:
    """Write a flat weight dict back into a model in-place."""
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())


def model_path(model_id: str) -> Path:
    """Resolve the local cache path for a model."""
    return Path(get_model_path(model_id))


def save_weights(weights: dict[str, mx.array], out_dir: str | Path) -> Path:
    """
    Save a flat weight dict as safetensors shards under out_dir.
    Returns the output directory path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    return out


def copy_config(model_id: str, out_dir: str | Path) -> None:
    """Copy tokenizer and config files from the source model into out_dir."""
    src = model_path(model_id)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.json", "*.model", "*.tiktoken", "tokenizer*"):
        for f in src.glob(pattern):
            shutil.copy2(f, dst / f.name)
