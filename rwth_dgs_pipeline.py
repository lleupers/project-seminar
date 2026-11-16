"""Compare ASL and DGS 1-NN references on the same held-out RWTH signers."""
import argparse
from pathlib import Path
from random import Random
from string import ascii_uppercase
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from autoencoder.model import HandAutoencoder
from components import kNN
from utils import LandmarkProcessing

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_data"
OUTPUTS = ROOT / "autoencoder" / "outputs"
RWTH_URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/I6_Gestures"
LABELS = {
    str(number): letter
    for number, letter in enumerate(ascii_uppercase, 1)
    if letter not in {"J", "Z"}
}


def download_if_missing(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:
        content = response.read()
    partial = path.with_suffix(".part")
    partial.write_bytes(content)
    partial.replace(path)


def load_clips():
    manifest = RAW / "rwth_index.txt"
    download_if_missing(f"{RWTH_URL}/i6_data_P20_BOTH.txt", manifest)
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sequence = Path(line.strip())
        gesture = sequence.parent.name
        if sequence.suffix != ".seq" or gesture not in LABELS:
            continue
        signer, _, session, camera = sequence.stem.split("_")
        rows.append({
            "sample_id": f"{gesture}/{sequence.with_suffix('.mpg').name}",
            "label": LABELS[gesture],
            "signer": int(signer),
            "session": int(session),
            "camera": camera,
        })
    clips = pd.DataFrame(rows)
    if len(clips) != 960 or clips.sample_id.duplicated().any() or clips.signer.nunique() != 20:
        raise ValueError("Expected 960 unique RWTH clips from 20 signers")
    return clips


@torch.inference_mode()
def encode_clip(path, model, processor):
    # Use the same central nine frames as rwth_dataset_testing.py.
    video = cv2.VideoCapture(str(path))
    poses = []
    try:
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise RuntimeError(f"No readable frames in {path}")
        indices = set(np.rint(np.linspace(0.25, 0.75, 9) * (frame_count - 1)).astype(int))
        for index in range(max(indices) + 1):
            ok, frame = video.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index} in {path}")
            if index in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose = processor.image2landmark(image)
                if pose.any() and torch.isfinite(pose).all():
                    poses.append(pose)
    finally:
        video.release()

    if not poses:
        return None
    return model.encode(torch.stack(poses)).mean(dim=0).numpy()


def main(output, seed):
    clips = load_clips()
    signers = sorted(clips.signer.unique().tolist())
    held_out = sorted(Random(seed).sample(signers, 4))
    clips["split"] = np.where(clips.signer.isin(held_out), "test", "train")
    print(f"Seed {seed}, test signers: {held_out}", flush=True)

    checkpoint = torch.load(OUTPUTS / "autoencoder.pt", map_location="cpu", weights_only=True)
    dim = checkpoint["latent_dim"]
    model = HandAutoencoder(latent_dim=dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    asl_reference = OUTPUTS / "train_latents.csv"
    asl_x, asl_y, asl_ids = kNN.load_latents(asl_reference)
    train_files = {Path(path).as_posix() for path in checkpoint["train_files"]}
    if asl_x.shape[1] != dim or {Path(path).as_posix() for path in asl_ids} != train_files:
        raise ValueError("ASL reference does not match the checkpoint training split")
    shared = set(LABELS.values())
    if set(asl_y) != shared:
        raise ValueError("ASL reference must contain exactly the shared classes")

    vectors = np.full((len(clips), dim), np.nan, dtype=np.float32)
    processor = LandmarkProcessing(str(ROOT / "mediapipe_models" / "hand_landmarker.task"))
    try:
        for index, sample_id in enumerate(tqdm(clips.sample_id, desc="RWTH", mininterval=5)):
            path = RAW / "rwth" / sample_id
            download_if_missing(f"{RWTH_URL}/video/{sample_id}", path)
            embedding = encode_clip(path, model, processor)
            if embedding is not None:
                vectors[index] = embedding
    finally:
        processor.landmarker.close()

    clips["has_embedding"] = np.isfinite(vectors).all(axis=1)
    columns = [f"z{i}" for i in range(dim)]
    latents = clips[["sample_id", "label"]].join(pd.DataFrame(vectors, columns=columns))
    train = clips.split.eq("train") & clips.has_embedding
    test = clips.split.eq("test")
    valid_test = test & clips.has_embedding
    if set(clips.loc[train, "label"]) != shared:
        raise ValueError("DGS reference is missing shared classes")

    output.mkdir(parents=True, exist_ok=True)
    clips.to_csv(output / "split.csv", index=False)
    dgs_reference = output / "dgs_train_latents.csv"
    latents.loc[train].to_csv(dgs_reference, index=False)
    latents.loc[valid_test].to_csv(output / "dgs_test_latents.csv", index=False)
    evaluate_references(output, seed, clips.loc[test], asl_reference, dgs_reference)


def evaluate_references(output, seed, test_clips, asl_reference, dgs_reference):
    predictions = test_clips.copy().set_index("sample_id")
    n_predicted = int(test_clips.has_embedding.sum())
    if n_predicted:
        x_test, y_test, test_ids = kNN.load_latents(output / "dgs_test_latents.csv")

    summaries = []
    for name, reference in (("ASL", asl_reference), ("DGS", dgs_reference)):
        report_dir = output / name.lower()
        model_path = kNN.train_knn_from_latents(
            reference, report_dir / "model.csv", k=3, metric="cosine"
        )
        column = f"{name.lower()}_prediction"
        predictions[column] = "NO_PREDICTION"
        if n_predicted:
            metrics = kNN.evaluate(model_path, x_test, y_test, test_ids, report_dir)
            report = pd.read_csv(report_dir / "predictions.csv").set_index("sample_id")
            predictions.loc[report.index, column] = report.predicted_label
            n_reference = metrics["n_reference"]
        else:
            _, metadata = kNN.load_model(model_path)
            n_reference = metadata["n_samples"]
        correct = predictions[column].eq(predictions.label)
        summaries.append({
            "reference": name,
            "seed": seed,
            "n_reference": n_reference,
            "n_test": len(predictions),
            "n_predicted": n_predicted,
            "correct": int(correct.sum()),
            "accuracy": correct.mean(),
        })

    predictions.reset_index().to_csv(output / "predictions.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "rwth_dgs")
    args = parser.parse_args()
    main(args.output, args.seed)
