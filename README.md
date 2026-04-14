# LLM Weight Corruption Framework

Bare-metal LLM weight corruption for Apple Silicon using [MLX](https://github.com/ml-explore/mlx).  
Directly manipulates model weight tensors — no servers, no middlemen.

---

## What it does

Loads an MLX-format language model, applies a corruption strategy to selected weight tensors, and saves a new model you can immediately load and test. A built-in benchmark runner lets you compare outputs and latency before and after corruption side-by-side.

Use cases:
- Studying model robustness to weight noise
- Simulating hardware faults / bit-flip errors
- Understanding which weight structures carry the most information
- Adversarial ML and safety research

---

## Setup

**Requirements:** macOS on Apple Silicon (M1/M2/M3/M4), Python 3.10+

```bash
git clone https://github.com/Phygineer/Weights-Corruption
cd Weights-Corruption

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the env template and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```env
MLX_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit   # HuggingFace repo or local path
CORRUPTION_SEED=42
CORRUPTION_RATE=0.01
CORRUPTION_MAGNITUDE=1.0
RESULTS_DIR=results/
```

Models are downloaded automatically from HuggingFace on first run and cached locally.  
Browse available MLX models at [huggingface.co/mlx-community](https://huggingface.co/mlx-community).

---

## Usage

### Interactive menu (recommended)

```bash
python scripts/run.py
```

Arrow-key menu walks you through model selection, strategy, target tensors, and parameters. No flags needed.

### Command-line flags

```bash
python scripts/run.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | from `.env` | HuggingFace repo ID or local path |
| `--strategy` | — | Corruption strategy (see below) |
| `--target` | `all` | Which tensors to corrupt: `all`, `attn`, `ffn` |
| `--rate` | `0.01` | Fraction of weights to corrupt (0.0–1.0) |
| `--magnitude` | `1.0` | Noise or scale magnitude |
| `--seed` | `42` | Random seed for reproducibility |
| `--num-heads` | `32` | Attention head count (used by `head` strategy) |
| `--output` | auto | Output directory for the corrupted model |
| `--info` | — | Print all tensor names and shapes, then exit |
| `--benchmark` | — | Run baseline + corrupted benchmark after corruption |

### Examples

```bash
# Interactive menu
python scripts/run.py

# Inspect what tensors are inside the model
python scripts/run.py --info

# Gaussian noise on all weights, then compare outputs
python scripts/run.py --strategy gaussian --rate 0.01 --benchmark

# Ablate 10% of attention heads (attn tensors only)
python scripts/run.py --strategy head --target attn --num-heads 32 --rate 0.10

# SVD corruption on FFN layers only
python scripts/run.py --strategy svd --target ffn --magnitude 2.0

# Zero-ablate 5% of all weights with a fixed seed
python scripts/run.py --strategy zero --rate 0.05 --seed 123

# Use a different model
python scripts/run.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --strategy signflip
```

---

## Corruption strategies

### Value-level

These treat each weight value independently.

| Strategy | Flag | Description |
|----------|------|-------------|
| **Gaussian noise** | `gaussian` | Add `N(0, magnitude)` noise to a random `rate` fraction of weight values. Soft degradation — model output drifts gradually. |
| **Zero ablation** | `zero` | Set a random `rate` fraction of weights to zero. Simulates dead neurons / synaptic pruning. |
| **Scale corruption** | `scale` | Multiply random weights by a uniform random factor in `[-magnitude, magnitude]`. Creates large outlier activations. |
| **Sign flip** | `signflip` | Flip the sign of a random `rate` fraction of weights. Reverses learned gradient directions. |

### Structured

These operate on the geometry of weight matrices rather than individual values.

| Strategy | Flag | Description |
|----------|------|-------------|
| **SVD corruption** | `svd` | Decomposes each 2-D weight matrix via SVD, adds noise to singular values proportional to their magnitude, then reconstructs. Destroys specific learned directions in weight space. Only 2-D matrices are affected. |
| **Head ablation** | `head` | Zeros out entire attention heads by chunking Q/K/V/O projection matrices into `num_heads` slices and nullifying `rate` fraction of them. The most targeted strategy — kills specific reasoning capabilities rather than random values. |

### Target tensor selection

All strategies can be scoped to a subset of tensors:

| `--target` | Affected layers |
|------------|----------------|
| `all` | All attention and FFN weight matrices (default) |
| `attn` | Q, K, V, O projections only |
| `ffn` | Gate, up, down projections (MLP blocks) only |

Normalization layers, biases, and embedding tables are never corrupted.

---

## Output

After corruption, a new model directory is written to `results/<model_name>_corrupted/` containing:
- `weights.safetensors` — the corrupted weights
- All tokenizer and config files copied from the source model

This directory can be loaded directly as a model:

```python
from mlx_lm import load, generate
model, tokenizer = load("results/mlx-community_Llama-3.2-3B-Instruct-4bit_corrupted")
print(generate(model, tokenizer, prompt="What is 2+2?", max_tokens=64))
```

Benchmark reports (JSON) are saved to `results/baseline.json` and `results/corrupted_<strategy>.json`.

---

## Project structure

```
Weights-Corruption/
├── scripts/
│   └── run.py                    # Entry point
├── src/
│   ├── cli.py                    # Interactive menu + argparse CLI
│   ├── config.py                 # Loads .env variables
│   ├── corruption/
│   │   ├── strategies.py         # All corruption strategies (mx.array)
│   │   └── engine.py             # Load → corrupt → save pipeline
│   ├── evaluation/
│   │   └── benchmarks.py         # MLX-native generation + reporting
│   └── models/
│       └── mlx_loader.py         # Model loading, weight access, saving
├── .env                          # Local config (gitignored)
├── .env.example                  # Template
└── requirements.txt
```

---

## How it works

1. **Load** — `mlx_lm.load()` pulls the model into unified memory as MLX arrays
2. **Flatten** — `mlx.utils.tree_flatten(model.parameters())` gives a flat `{name: mx.array}` dict
3. **Filter** — tensor names are matched against the chosen target (`all` / `attn` / `ffn`)
4. **Corrupt** — the strategy's `apply()` transforms each selected `mx.array` in-place
5. **Eval** — `mx.eval()` materialises all lazy computations
6. **Save** — `mx.save_safetensors()` writes the corrupted weights; tokenizer/config files are copied alongside

The original model cache is never modified.

---

## Notes

- MLX runs on Apple Silicon only (Metal GPU / unified memory)
- Quantized models (4-bit, 8-bit) are dequantized to float32 before corruption and saved as float32 — the output file will be larger than the source
- `--strategy svd` is significantly slower than the others due to per-tensor SVD decomposition on CPU
- `--strategy head` requires knowing the model's attention head count (`--num-heads`); check the model's `config.json` if unsure
