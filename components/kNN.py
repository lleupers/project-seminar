from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

META_PREFIX = "# knn_meta: "

# Loads latent vectors and labels from a CSV file. Checks for missing values etc.
def load_latents(csv_path, id_col="sample_id", label_col="label", z_prefix="z"):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Latent file not found: {csv_path}")
 
    with open(csv_path, encoding="utf-8") as f:
        has_meta = f.readline().startswith(META_PREFIX)
    df = pd.read_csv(csv_path, skiprows=1 if has_meta else 0)
 
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
 
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
    y = df[label_col].astype(str).to_numpy()
    ids = df[id_col].astype(str).to_numpy()
    return X, y, ids


# Trains a kNN model with the given latent vectors and labels.
# X is a matric of latent vectors, y is a vector of labels.
def train_knn(X, y, k=1, metric="cosine", weights="uniform"):
    if k < 1:
        raise ValueError(f"k={k} must be at least 1")
    
    if k > len(X):
        raise ValueError(f"k={k} is greater than the number of samples {len(X)}")

    model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights = weights, algorithm = "brute")
    model.fit(X, y)
    return model

def save_model(model_path, X, y, ids, k, metric, weights):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
 
    df = pd.DataFrame(X, columns=[f"z{i}" for i in range(X.shape[1])])
    df.insert(0, "label", y)
    df.insert(0, "sample_id", ids)
 
    meta = {"k": k, "metric": metric, "weights": weights, "dim": int(X.shape[1])}
    with open(model_path, "w", encoding="utf-8", newline="") as f:
        f.write(META_PREFIX + json.dumps(meta) + "\n")
        df.to_csv(f, index=False)
    return model_path

# Loads a saved model CSV and returns a fitted kNN plus its metadata.
def load_model(model_path, k=None):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"kNN model not found: {model_path}. Run train_knn_from_latents first.")
 
    with open(model_path, encoding="utf-8") as f:
        first = f.readline().strip()
    if not first.startswith(META_PREFIX):
        raise ValueError(f"{model_path} is not a kNN model file (missing '{META_PREFIX.strip()}' header)")
    meta = json.loads(first[len(META_PREFIX):])
 
    X, y, ids = load_latents(model_path)
    if X.shape[1] != meta["dim"]:
        raise ValueError(f"Model file inconsistent: {X.shape[1]} latent columns, header says {meta['dim']}")
 
    if k is not None:
        meta["k"] = k
    model = train_knn(X, y, k=meta["k"], metric=meta["metric"], weights=meta["weights"])
    meta["n_samples"] = len(X)
    return model, meta

# Takes the encoder output (one or several CSVs), fits the kNN and saves it as CSV.
def train_knn_from_latents(latent_paths, model_path, k=1, metric="cosine", weights="uniform", append=False):
    if isinstance(latent_paths, (str, Path)):
        latent_paths = [latent_paths]
 
    Xs, ys, idss = [], [], []
    for p in latent_paths:
        X, y, ids = load_latents(p)
        if Xs and X.shape[1] != Xs[0].shape[1]:
            raise ValueError(f"{p} has {X.shape[1]} latent dims, previous files have {Xs[0].shape[1]}")
        Xs.append(X); ys.append(y); idss.append(ids)
 
    if append and Path(model_path).is_file():
        _, meta = load_model(model_path)
        X_old, y_old, ids_old = load_latents(model_path)
        if X_old.shape[1] != Xs[0].shape[1]:
            raise ValueError(
                f"New latents have {Xs[0].shape[1]} dims, model has {X_old.shape[1]}. "
                "Was the autoencoder retrained? Then rebuild the model from scratch."
            )
        Xs.insert(0, X_old); ys.insert(0, y_old); idss.insert(0, ids_old)
        k, metric, weights = meta["k"], meta["metric"], meta["weights"]
 
    X, y, ids = np.vstack(Xs), np.concatenate(ys), np.concatenate(idss)
 
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs across the given files / existing model")
 
    train_knn(X, y, k=k, metric=metric, weights=weights)
    save_model(model_path, X, y, ids, k, metric, weights)
 
    classes, counts = np.unique(y, return_counts=True)
    print(f"[kNN] Saved model to {model_path}: {len(X)} vectors, {len(classes)} classes, "
          f"dim={X.shape[1]}, k={k}, metric={metric}")
    if (counts < k).any():
        print(f"[kNN] Warning: classes with fewer than k={k} samples: {list(classes[counts < k])}")
    return model_path


# Predicts the label for a given latent vector.
def predict(model, latent_vector):
    x = np.asarray(latent_vector, dtype="float32").reshape(1, -1)
    return model.predict(x)[0]


    # Predicts and evaluates the model on a test set.
def evaluate(model_path, X_test, y_test, ids_test, out_dir, k=None):
    model, meta = load_model(model_path, k=k)
 
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    y_pred = model.predict(X_test)
 
    pred_df = pd.DataFrame({
        "sample_id": ids_test,
        "true_label": y_test,
        "predicted_label": y_pred,
        "correct": y_pred == y_test,
    })
 
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
 
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "per_class_metrics.csv")
 
    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")
 
    summary = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "n_test": len(y_test),
        "k": meta["k"],
        "metric": meta["metric"],
        "n_reference": meta["n_samples"],
        "model_path": str(model_path),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    return summary
