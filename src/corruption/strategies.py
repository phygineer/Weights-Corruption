"""
Corruption strategies — all operate directly on mx.array weight tensors.

Available strategies:
  gaussian   — add Gaussian noise to a random subset of values
  zero       — zero out a random fraction of weights
  scale      — multiply random weights by a large random factor
  signflip   — flip the sign of random weights
  svd        — corrupt singular values (destroys weight-space directions)
  head       — zero out entire attention heads (structured ablation)
"""

import mlx.core as mx

from src.config import CORRUPTION_SEED, CORRUPTION_RATE, CORRUPTION_MAGNITUDE


class CorruptionStrategy:
    description: str = ""

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Value-level strategies
# ---------------------------------------------------------------------------

class GaussianNoise(CorruptionStrategy):
    """Add Gaussian noise to a random subset of weight values."""
    description = "Add Gaussian noise to randomly selected weight values"

    def __init__(self, rate: float = CORRUPTION_RATE, magnitude: float = CORRUPTION_MAGNITUDE, seed: int = CORRUPTION_SEED):
        self.rate = rate
        self.magnitude = magnitude
        self.seed = seed

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        mx.random.seed(self.seed)
        mask = mx.random.uniform(shape=weight.shape) < self.rate
        noise = mx.random.normal(shape=weight.shape) * self.magnitude
        return weight + mask * noise


class ZeroAblation(CorruptionStrategy):
    """Zero out a random fraction of weights (simulates dead neurons)."""
    description = "Zero out a random fraction of weights"

    def __init__(self, rate: float = CORRUPTION_RATE, seed: int = CORRUPTION_SEED):
        self.rate = rate
        self.seed = seed

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        mx.random.seed(self.seed)
        mask = mx.random.uniform(shape=weight.shape) >= self.rate
        return weight * mask


class ScaleCorruption(CorruptionStrategy):
    """Multiply random weights by a large random factor."""
    description = "Multiply random weights by a large random factor"

    def __init__(self, rate: float = CORRUPTION_RATE, magnitude: float = CORRUPTION_MAGNITUDE, seed: int = CORRUPTION_SEED):
        self.rate = rate
        self.magnitude = magnitude
        self.seed = seed

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        mx.random.seed(self.seed)
        mask = mx.random.uniform(shape=weight.shape) < self.rate
        scale = mx.random.uniform(shape=weight.shape, low=-self.magnitude, high=self.magnitude)
        return mx.where(mask, weight * scale, weight)


class SignFlip(CorruptionStrategy):
    """Flip the sign of a random fraction of weights."""
    description = "Flip the sign of randomly selected weights"

    def __init__(self, rate: float = CORRUPTION_RATE, seed: int = CORRUPTION_SEED):
        self.rate = rate
        self.seed = seed

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        mx.random.seed(self.seed)
        mask = mx.random.uniform(shape=weight.shape) < self.rate
        return mx.where(mask, -weight, weight)


# ---------------------------------------------------------------------------
# Structured strategies
# ---------------------------------------------------------------------------

class SVDCorruption(CorruptionStrategy):
    """
    Decompose each weight matrix via SVD, corrupt singular values, reconstruct.
    Destroys specific learned directions in weight space rather than random scalars.
    Only applied to 2-D weight matrices.
    """
    description = "Corrupt singular values via SVD (destroys learned weight-space directions)"

    def __init__(self, rate: float = CORRUPTION_RATE, magnitude: float = CORRUPTION_MAGNITUDE, seed: int = CORRUPTION_SEED):
        self.rate = rate
        self.magnitude = magnitude
        self.seed = seed

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        if weight.ndim != 2:
            return weight  # skip non-matrix tensors

        mx.random.seed(self.seed)
        U, S, Vt = mx.linalg.svd(weight, full_matrices=False, stream=mx.cpu)

        mask = mx.random.uniform(shape=S.shape) < self.rate
        noise = mx.random.normal(shape=S.shape) * self.magnitude * mx.mean(mx.abs(S))
        S_corrupted = S + mask * noise

        # Reconstruct: U @ diag(S) @ Vt — MLX lacks mx.diag so we broadcast
        return (U * S_corrupted[None, :]) @ Vt


class HeadAblation(CorruptionStrategy):
    """
    Zero out entire attention heads by chunking the weight matrix along its
    first dimension into num_heads slices and nullifying selected ones.
    Only meaningful for Q/K/V/O projection tensors.
    """
    description = "Zero out entire attention heads (structured, not random)"

    def __init__(self, num_heads: int = 32, head_rate: float = 0.1, seed: int = CORRUPTION_SEED):
        self.num_heads = num_heads
        self.head_rate = head_rate  # fraction of heads to kill
        self.seed = seed

    def apply(self, weight: mx.array, name: str = "") -> mx.array:
        proj_keywords = ("q_proj", "k_proj", "v_proj", "o_proj")
        if not any(kw in name for kw in proj_keywords):
            return weight

        rows = weight.shape[0]
        head_dim = rows // self.num_heads
        if head_dim == 0 or rows % self.num_heads != 0:
            return weight  # shape doesn't match — skip

        mx.random.seed(self.seed)
        n_kill = max(1, int(self.num_heads * self.head_rate))
        kill_indices = set(mx.random.permutation(self.num_heads)[:n_kill].tolist())

        parts = []
        for h in range(self.num_heads):
            if h in kill_indices:
                parts.append(mx.zeros((head_dim, *weight.shape[1:])))
            else:
                parts.append(weight[h * head_dim:(h + 1) * head_dim])
        return mx.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[CorruptionStrategy]] = {
    "gaussian":  GaussianNoise,
    "zero":      ZeroAblation,
    "scale":     ScaleCorruption,
    "signflip":  SignFlip,
    "svd":       SVDCorruption,
    "head":      HeadAblation,
}

STRATEGY_DESCRIPTIONS = {k: v.description for k, v in STRATEGIES.items()}


def get_strategy(name: str, **kwargs) -> CorruptionStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)
