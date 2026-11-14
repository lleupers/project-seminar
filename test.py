import components.kNN as kNN
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "autoencoder" / "outputs"

# Generates unique run ID based on the current timestamp.
def make_run_id(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    return datetime.now().strftime(fmt)

def main():
    X_test, y_test, ids_test = kNN.load_latents(OUTPUT_DIR / "test_latents.csv")
    results = OUTPUT_DIR / "knn_results" / make_run_id("%Y-%m-%d_%H-%M-%S_%f")
    model_path = kNN.train_knn_from_latents(
        OUTPUT_DIR / "train_latents.csv", results / "model.csv"
    )
    metrics = kNN.evaluate(model_path, X_test, y_test, ids_test, results)
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"Macro-F1: {metrics['macro_f1']:.1%}")
    print(f"Ergebnisse: {results}")


if __name__ == "__main__":
    main()
