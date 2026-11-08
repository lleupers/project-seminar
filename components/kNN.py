from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# Loads latent vectors and labels from a CSV file. Checks for missing values etc.
def load_latents (csv_path, id_col="sample_id", label_col="label", z_prefix="z"):
    df = pd.read_csv(csv_path)

    missing = {id_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    z_cols = [c for c in df.columns if c.startswith(z_prefix) and c[len(z_prefix):].isdigit()]
    z_cols.sort(key=lambda c: int(c[len(z_prefix):]))
    if not z_cols:
        raise ValueError(f"No latent columns found in {csv_path} with prefix '{z_prefix}'")

    if df[z_cols].isna().any().any():
        raise ValueError(f"NaN values found in latent columns of {csv_path}")
    if df[id_col].duplicated().any():
        raise ValueError(f"Duplicate sample IDs found in {csv_path}")

    X = df[z_cols].to_numpy(dtype="float32")
    y = df[label_col].to_numpy()
    ids = df[id_col].to_numpy()
    return X, y, ids

# Trains a kNN model with the given latent vectors and labels.
# X is a matric of latent vectors, y is a vector of labels.
def train_knn(X, y, k=1, metric="cosine", weights="uniform"):
    if k > len(X):
        raise ValueError(f"k={k} is greater than the number of samples {len(X)}")

    model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights = weights, algorithm = "brute")
    model.fit(X, y)
    return model

# Predicts the label for a given latent vector.
def predict(model, latent_vector):
    x = np.asarray(latent_vector, dtype="float32").reshape(1, -1)
    return model.predict(x)[0]

# Predicts and evaluates the model on a test set.
def evaluate(model, X_test, y_test, ids_test, y_train, ids_train, out_dir, top_n=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    top_n = min(top_n, len(y_train))
    distances, indices = model.kneighbors(X_test, n_neighbors=top_n)

    pred_df = pd.DataFrame({
        "sample_id": ids_test,
        "true_label": y_test,
        "predicted_label": y_pred,
        "correct": y_pred == y_test,
        "nn_distance": distances[:,0],
        "neighbor_labels": ["|".join(map(str, y_train[row])) for row in indices],
        "neighbor_ids": ["|".join(map(str, ids_train[row])) for row in indices],
    })

    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "per_class_metrics.csv")

    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "n_test": len(y_test),
    }