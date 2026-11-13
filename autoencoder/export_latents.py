import csv
from pathlib import Path

import torch

from .dataset import PoseDataset
from .model import HandAutoencoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def label_from_path(path):
    stem = Path(path).stem
    sample_number, separator, label = stem.partition("-")
    if separator and sample_number.isdecimal() and label:
        return label

    label, separator, sample_number = stem.rpartition("_")
    if separator and label and sample_number.isdecimal():
        return label

    return "UNBEKANNT"


def export_files(model, files, output_path, latent_dim, data_dir=DATA_DIR):
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["sample_id", "label"] + [f"z{i}" for i in range(latent_dim)]
        )

        with torch.no_grad():
            for relative_path in files:
                pose_path = data_dir / relative_path

                try:
                    pose = PoseDataset.load_pose(pose_path)
                except Exception as error:
                    print(f"Überspringe {relative_path}: {error}")
                    continue

                label = label_from_path(pose_path)
                latent = model.encode(pose)

                writer.writerow(
                    [relative_path, label] + latent.tolist()
                )
                count += 1

    print(f"Fertig: {count} Latent-Vektoren gespeichert in {output_path}")


def main(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "autoencoder.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    latent_dim = checkpoint["latent_dim"]

    model = HandAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Modell aus Epoche {checkpoint['epoch']} geladen."
    )

    print(
        f"Validierungsfehler: "
        f"{checkpoint['val_loss']:.6f}"
    )

    split_files = {split: checkpoint[f"{split}_files"] for split in ("train", "val")}
    test_files = [
        path.relative_to(data_dir).as_posix()
        for path in sorted((data_dir / "test").rglob("*.pt"))
    ]
    if test_files:
        used_files = set(split_files["train"]) | set(split_files["val"])
        if used_files.intersection(test_files):
            raise ValueError(
                "Für den Testexport bitte den Autoencoder mit dem neuen Split neu trainieren."
            )
        split_files["test"] = test_files

    for split, files in split_files.items():
        export_files(
            model,
            files,
            output_dir / f"{split}_latents.csv",
            latent_dim,
            data_dir=data_dir,
        )


if __name__ == "__main__":
    main()
