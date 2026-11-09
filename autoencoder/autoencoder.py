from pathlib import Path

from .export_latents import main as run_export
from .train import train_autoencoder


BASE_DIR = Path(__file__).resolve().parent


def run_pipeline():
    train_autoencoder(BASE_DIR.parent / "data", BASE_DIR / "outputs")
    run_export()


if __name__ == "__main__":
    run_pipeline()
