from pathlib import Path

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PoseDataset(Dataset):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = PROJECT_ROOT / "data"

        data_dir = Path(data_dir)

        # Findet auch Dateien in Unterordnern.
        candidates = sorted(data_dir.rglob("*.pt"))

        self.files = []

        # Dateien einmal prüfen, bevor das Training startet.
        for path in candidates:
            try:
                self.load_pose(path)
            except Exception as error:
                print(
                    f"Überspringe {path.name}: "
                    f"{type(error).__name__}: {error}"
                )
                continue

            self.files.append(path)

        if not self.files:
            raise ValueError(
                f"Keine gültigen .pt-Dateien in {data_dir} gefunden."
            )

        print(f"{len(self.files)} ladbare Posen gefunden.")

    @staticmethod
    def load_pose(path):
        pose = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(pose, torch.Tensor):
            raise ValueError("Datei enthält keinen Tensor.")

        if pose.shape != (21, 3):
            raise ValueError(
                f"Erwartet (21, 3), erhalten {pose.shape}."
            )

        pose = pose.float()

        if not torch.isfinite(pose).all():
            raise ValueError("Pose enthält ungültige Zahlen.")

        return pose

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.load_pose(self.files[index])