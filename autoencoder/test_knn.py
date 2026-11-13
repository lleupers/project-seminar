from pathlib import Path

import components.kNN as knn


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    X_train, y_train, ids_train = knn.load_latents(
        OUTPUT_DIR / "train_latents.csv"
    )

    X_val, y_val, ids_val = knn.load_latents(
        OUTPUT_DIR / "val_latents.csv"
    )

    print("Trainingsvektoren:", X_train.shape)
    print("Validierungsvektoren:", X_val.shape)

    # Funktioniert auch bei weniger als drei Trainingsposen.
    k = min(3, len(X_train))

    classifier = knn.train_knn(
        X_train,
        y_train,
        k=k,
    )

    prediction = knn.predict(
        classifier,
        X_val[0],
    )

    print("Vorhersage für die erste Validierungspose:", prediction)

    metrics = knn.evaluate(
        classifier,
        X_val,
        y_val,
        ids_val,
        y_train,
        ids_train,
        out_dir=OUTPUT_DIR / "knn_results",
        top_n=3,
    )

    print("Ergebnisse:", metrics)


if __name__ == "__main__":
    main()
