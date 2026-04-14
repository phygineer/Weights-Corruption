"""
Benchmark model quality before and after weight corruption.
Uses mlx-lm directly for generation — no server needed.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mlx_lm import generate, load as mlx_load

from src.config import RESULTS_DIR


DEFAULT_PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of France.",
    "Explain gravity in one sentence.",
    "Write a one-sentence story about a robot.",
    "What is the largest planet in the solar system?",
]


@dataclass
class EvalResult:
    prompt: str
    response: str
    latency_ms: float
    label: str = "baseline"


@dataclass
class BenchmarkReport:
    model: str
    strategy: str | None
    label: str
    results: list[EvalResult] = field(default_factory=list)

    def summary(self) -> dict:
        avg_latency = sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0
        return {
            "model": self.model,
            "strategy": self.strategy,
            "label": self.label,
            "num_prompts": len(self.results),
            "avg_latency_ms": round(avg_latency, 2),
        }


def run_benchmark(
    model_path: str,
    label: str = "baseline",
    strategy_name: str | None = None,
    prompts: list[str] = DEFAULT_PROMPTS,
    max_tokens: int = 128,
) -> BenchmarkReport:
    """
    Load the model at *model_path* and generate responses for each prompt.
    Works with the original model ID or a corrupted model directory.
    """
    print(f"[INFO] Benchmarking '{model_path}' ({label})...")
    model, tokenizer = mlx_load(model_path)
    report = BenchmarkReport(model=model_path, strategy=strategy_name, label=label)

    for prompt in prompts:
        start = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        report.results.append(EvalResult(prompt=prompt, response=response, latency_ms=elapsed, label=label))
        print(f"  [{elapsed:6.0f}ms] {prompt!r:.50s} → {response[:60]!r}")

    return report


def save_report(report: BenchmarkReport, filename: str) -> str:
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump({"summary": report.summary(), "results": [asdict(r) for r in report.results]}, f, indent=2)
    return str(path)


def compare_reports(baseline: BenchmarkReport, corrupted: BenchmarkReport) -> None:
    """Print a side-by-side latency and response diff."""
    print("\n--- Comparison ---")
    for b, c in zip(baseline.results, corrupted.results):
        print(f"\nPrompt : {b.prompt}")
        print(f"Before : {b.response[:120]}")
        print(f"After  : {c.response[:120]}")
        print(f"Latency: {b.latency_ms:.0f}ms → {c.latency_ms:.0f}ms")
