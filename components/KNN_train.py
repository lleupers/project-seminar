from pathlib import Path
import kNN
from datetime import datetime

# Generates unique run ID based on the current timestamp.
def make_run_id(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    return datetime.now().strftime(fmt)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

train_latent_vector_path = ...
model_path = ...

kNN.train_knn_from_latents(train_latent_vector_path, model_path)