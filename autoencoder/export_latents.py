import csv
from pathlib import Path

import torch

from .dataset import PoseDataset
from .model import HandAutoencoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def export_files(model, files, output_path, latent_dim):
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["sample_id", "label"] + [f"z{i}" for i in range(latent_dim)]
        )

        with torch.no_grad():
            for relative_path in files:
                pose_path = DATA_DIR / relative_path
                
                try:
                    pose = PoseDataset.load_pose(pose_path)
                except Exception as error:
                    print(f"Überspringe {relative_path}: {error}")
                    continue

                filename = Path(relative_path).name
                label = filename.split("_")[0] if "_" in filename else "UNBEKANNT"

                latent = model.encode(pose)

                writer.writerow(
                    [relative_path, label] + latent.tolist()
                )
                count += 1

    print(f"Fertig: {count} Latent-Vektoren gespeichert in {output_path}")


def main():
    checkpoint_path = OUTPUT_DIR / "autoencoder.pt"

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

    for split in ("train", "val"):
        files = checkpoint[f"{split}_files"]

        export_files(
            model,
            files,
            OUTPUT_DIR / f"{split}_latents.csv",
            latent_dim,
        )


if __name__ == "__main__":
    main()
