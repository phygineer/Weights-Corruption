from dotenv import load_dotenv
import os

load_dotenv()

MLX_MODEL = os.getenv("MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")

CORRUPTION_SEED = int(os.getenv("CORRUPTION_SEED", 42))
CORRUPTION_RATE = float(os.getenv("CORRUPTION_RATE", 0.01))
CORRUPTION_MAGNITUDE = float(os.getenv("CORRUPTION_MAGNITUDE", 1.0))

RESULTS_DIR = os.getenv("RESULTS_DIR", "results/")
