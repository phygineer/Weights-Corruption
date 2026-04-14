"""
Interactive CLI for the LLM Weight Corruption Framework.

Run interactively:   python scripts/run.py
Run with flags:      python scripts/run.py --strategy gaussian --benchmark
Help:                python scripts/run.py --help
"""

import argparse
import sys
from pathlib import Path

import questionary
from questionary import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from src import config
from src.corruption.engine import corrupt_model
from src.corruption.strategies import STRATEGIES, STRATEGY_DESCRIPTIONS, get_strategy
from src.evaluation.benchmarks import compare_reports, run_benchmark, save_report
from src.models.mlx_loader import get_weights, load_model

console = Console()

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

Q_STYLE = Style([
    ("qmark",        "fg:#00d7ff bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00ff87 bold"),
    ("pointer",      "fg:#ff5f87 bold"),
    ("highlighted",  "fg:#ff5f87 bold"),
    ("selected",     "fg:#00ff87"),
    ("separator",    "fg:#444444"),
    ("instruction",  "fg:#888888"),
])

STRATEGY_CHOICES = [
    questionary.Choice(
        title=f"{name:<10} — {STRATEGY_DESCRIPTIONS[name]}",
        value=name,
    )
    for name in STRATEGIES
]

TARGET_CHOICES = [
    questionary.Choice("All weight matrices (attn + FFN)",  value="all"),
    questionary.Choice("Attention only  (q/k/v/o_proj)",   value="attn"),
    questionary.Choice("FFN / MLP only  (gate/up/down)",   value="ffn"),
]

LAYER_KEYWORDS = {
    "all":  ("attn", "mlp", "ffn", "proj", "fc", "gate", "up", "down"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "ffn":  ("gate_proj", "up_proj", "down_proj", "fc1", "fc2", "mlp"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold magenta]LLM Weight Corruption Framework[/bold magenta]\n"
            "[dim]Bare Metal · MLX Edition[/dim]",
            justify="center",
        ),
        border_style="magenta",
        padding=(1, 4),
    ))
    console.print()


def _make_tensor_filter(target: str):
    keywords = LAYER_KEYWORDS[target]

    def _filter(name: str) -> bool:
        n = name.lower()
        return any(kw in n for kw in keywords) and "norm" not in n

    return _filter


def _strategy_kwargs(name: str, rate: float, magnitude: float, seed: int, num_heads: int) -> dict:
    if name == "head":
        return {"num_heads": num_heads, "head_rate": rate, "seed": seed}
    if name in ("zero", "signflip"):
        return {"rate": rate, "seed": seed}
    return {"rate": rate, "magnitude": magnitude, "seed": seed}


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_info(model_id: str) -> None:
    console.rule("[bold cyan]Tensor Inspector[/bold cyan]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
        task = p.add_task(f"Loading [cyan]{model_id}[/cyan]…", total=None)
        model, _ = load_model(model_id)
        weights = get_weights(model)
        p.update(task, completed=True)

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("Tensor", style="white")
    table.add_column("Shape", style="yellow", justify="right")
    table.add_column("Type", style="dim")

    for name, arr in sorted(weights.items()):
        table.add_row(name, str(list(arr.shape)), str(arr.dtype))

    console.print(table)
    console.print(f"[dim]Total: {len(weights)} tensors[/dim]\n")


def cmd_corrupt(model_id: str, strategy_name: str, rate: float, magnitude: float,
                seed: int, num_heads: int, target: str, output: str | None) -> Path:
    console.rule("[bold red]Corrupting Weights[/bold red]")

    kwargs = _strategy_kwargs(strategy_name, rate, magnitude, seed, num_heads)
    strategy = get_strategy(strategy_name, **kwargs)
    tensor_filter = _make_tensor_filter(target)

    table = Table(box=box.ROUNDED, show_header=False, border_style="dim")
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bold yellow")
    table.add_row("Model",    model_id)
    table.add_row("Strategy", strategy_name)
    table.add_row("Target",   target)
    table.add_row("Rate",     str(rate))
    if strategy_name not in ("zero", "signflip", "head"):
        table.add_row("Magnitude", str(magnitude))
    if strategy_name == "head":
        table.add_row("Num heads", str(num_heads))
    table.add_row("Seed",     str(seed))
    console.print(table)
    console.print()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
        p.add_task("Corrupting weights…", total=None)
        out = corrupt_model(model_id=model_id, strategy=strategy,
                            tensor_filter=tensor_filter, output_dir=output)

    console.print(f"\n[bold green]Done![/bold green] Corrupted model saved to [cyan]{out}[/cyan]\n")
    return out


def cmd_benchmark(model_id: str, corrupted_dir: Path | None = None,
                  strategy_name: str | None = None) -> None:
    console.rule("[bold blue]Benchmark[/bold blue]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
        p.add_task("Running baseline…", total=None)
        baseline = run_benchmark(model_id, label="baseline")
    save_report(baseline, "baseline.json")

    corrupted = None
    if corrupted_dir:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
            p.add_task("Running corrupted model…", total=None)
            corrupted = run_benchmark(str(corrupted_dir), label="corrupted",
                                      strategy_name=strategy_name)
        save_report(corrupted, f"corrupted_{strategy_name or 'unknown'}.json")

    # Summary table
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold blue")
    table.add_column("Prompt",          style="white",  max_width=40)
    table.add_column("Baseline",        style="green",  max_width=40)
    table.add_column("Corrupted",       style="red",    max_width=40)
    table.add_column("Latency Δ",       style="yellow", justify="right")

    for i, b in enumerate(baseline.results):
        c_resp = corrupted.results[i].response if corrupted else "[dim]—[/dim]"
        c_lat  = corrupted.results[i].latency_ms if corrupted else 0
        delta  = f"{c_lat - b.latency_ms:+.0f}ms" if corrupted else "—"
        table.add_row(
            b.prompt[:38],
            b.response[:38],
            c_resp[:38],
            delta,
        )

    console.print(table)
    console.print(f"\n[dim]Reports saved to [cyan]{config.RESULTS_DIR}[/cyan][/dim]\n")


# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------

def _interactive() -> None:
    _banner()

    action = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Corrupt model weights",         value="corrupt"),
            questionary.Choice("Inspect model tensors",         value="info"),
            questionary.Choice("Benchmark (baseline only)",     value="bench_base"),
            questionary.Choice("Corrupt + benchmark (compare)", value="corrupt_bench"),
            questionary.Separator(),
            questionary.Choice("Exit",                          value="exit"),
        ],
        style=Q_STYLE,
    ).ask()

    if action is None or action == "exit":
        console.print("[dim]Bye.[/dim]")
        sys.exit(0)

    model_id = questionary.text(
        "Model (HuggingFace repo ID or local path):",
        default=config.MLX_MODEL,
        style=Q_STYLE,
    ).ask()

    if action == "info":
        cmd_info(model_id)
        return

    if action == "bench_base":
        cmd_benchmark(model_id)
        return

    # --- corruption path ---
    strategy_name = questionary.select(
        "Corruption strategy:",
        choices=STRATEGY_CHOICES,
        style=Q_STYLE,
    ).ask()

    target = questionary.select(
        "Which tensors to target?",
        choices=TARGET_CHOICES,
        style=Q_STYLE,
    ).ask()

    rate = float(questionary.text(
        "Corruption rate (0.0 – 1.0):",
        default=str(config.CORRUPTION_RATE),
        validate=lambda v: 0 < float(v) <= 1 if _is_float(v) else "Enter a number between 0 and 1",
        style=Q_STYLE,
    ).ask())

    magnitude = config.CORRUPTION_MAGNITUDE
    num_heads = 32

    if strategy_name not in ("zero", "signflip", "head"):
        magnitude = float(questionary.text(
            "Magnitude (noise scale):",
            default=str(config.CORRUPTION_MAGNITUDE),
            validate=lambda v: _is_float(v) or "Enter a number",
            style=Q_STYLE,
        ).ask())

    if strategy_name == "head":
        num_heads = int(questionary.text(
            "Number of attention heads in the model:",
            default="32",
            validate=lambda v: v.isdigit() or "Enter an integer",
            style=Q_STYLE,
        ).ask())

    seed = int(questionary.text(
        "Random seed:",
        default=str(config.CORRUPTION_SEED),
        validate=lambda v: v.isdigit() or "Enter an integer",
        style=Q_STYLE,
    ).ask())

    out = cmd_corrupt(model_id, strategy_name, rate, magnitude, seed, num_heads, target, None)

    if action == "corrupt_bench":
        cmd_benchmark(model_id, corrupted_dir=out, strategy_name=strategy_name)


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Argument-based flow
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "LLM Weight Corruption Framework — MLX Edition\n\n"
            "Run without arguments for the interactive menu.\n"
            "Pass flags to skip the menu and run directly.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
STRATEGIES
  gaussian   Add Gaussian noise to random weight values
  zero       Zero out a random fraction of weights
  scale      Multiply random weights by a random factor
  signflip   Flip the sign of random weights
  svd        Corrupt singular values (breaks weight-space directions)
  head       Zero out entire attention heads (structured ablation)

TARGET TENSORS
  all        All weight matrices (attention + FFN)
  attn       Attention projections only (q/k/v/o_proj)
  ffn        FFN / MLP layers only (gate/up/down_proj)

EXAMPLES
  # Interactive menu
  python scripts/run.py

  # List all tensors
  python scripts/run.py --info

  # Corrupt with Gaussian noise, then benchmark
  python scripts/run.py --strategy gaussian --rate 0.01 --benchmark

  # Ablate 10% of attention heads, target attn tensors only
  python scripts/run.py --strategy head --target attn --num-heads 32 --rate 0.10

  # SVD corruption on FFN layers
  python scripts/run.py --strategy svd --target ffn --magnitude 2.0
        """,
    )

    parser.add_argument("--model",      default=config.MLX_MODEL,       help="HuggingFace repo ID or local path")
    parser.add_argument("--strategy",   choices=list(STRATEGIES),        help="Corruption strategy")
    parser.add_argument("--target",     choices=["all","attn","ffn"],    help="Which tensors to corrupt (default: all)", default="all")
    parser.add_argument("--rate",       type=float, default=config.CORRUPTION_RATE,      help="Fraction of weights to corrupt (0.0–1.0)")
    parser.add_argument("--magnitude",  type=float, default=config.CORRUPTION_MAGNITUDE, help="Noise / scale magnitude")
    parser.add_argument("--seed",       type=int,   default=config.CORRUPTION_SEED,      help="Random seed")
    parser.add_argument("--num-heads",  type=int,   default=32,          help="Attention head count (for --strategy head)")
    parser.add_argument("--output",     default=None,                    help="Output directory for corrupted model")
    parser.add_argument("--info",       action="store_true",             help="Print tensor names/shapes and exit")
    parser.add_argument("--benchmark",  action="store_true",             help="Benchmark original + corrupted after corruption")

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        _interactive()
        return

    parser = _build_parser()
    args = parser.parse_args()

    _banner()

    if args.info:
        cmd_info(args.model)
        return

    if not args.strategy:
        console.print("[bold red]Error:[/bold red] --strategy is required when running non-interactively.")
        console.print(f"Choices: {', '.join(STRATEGIES)}\n")
        parser.print_usage()
        sys.exit(1)

    out = cmd_corrupt(
        model_id=args.model,
        strategy_name=args.strategy,
        rate=args.rate,
        magnitude=args.magnitude,
        seed=args.seed,
        num_heads=args.num_heads,
        target=args.target,
        output=args.output,
    )

    if args.benchmark:
        cmd_benchmark(args.model, corrupted_dir=out, strategy_name=args.strategy)
