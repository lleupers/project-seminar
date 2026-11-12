import components.kNN as kNN
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Generates unique run ID based on the current timestamp.
def make_run_id(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    return datetime.now().strftime(fmt)

# 1. Data preparation

# 2. Pose estimation

# 3. Autoencoding (inference)

# 4. kNN classification (inference)
test_latent_vector_path = ...
model_path = ...

X_test, y_test, ids_test = kNN.load_latents(test_latent_vector_path)
knn_path = OUTPUT_DIR / "results" / "knn_" / make_run_id()
kNN.evaluate(model_path, X_test, y_test, ids_test, knn_path)