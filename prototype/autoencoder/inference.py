import argparse
import json
from pathlib import Path

import torch

from .dataset import PoseDataset
from .model import HandAutoencoder


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "outputs" / "autoencoder.pt"
DEFAULT_HAND_MODEL = BASE_DIR.parent / "mediapipe_models" / "hand_landmarker.task"


def load_autoencoder(checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = HandAutoencoder(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.inference_mode()
def encode_input(model, input_path, hand_model_path=DEFAULT_HAND_MODEL):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    if input_path.suffix.lower() == ".pt":
        pose = PoseDataset.load_pose(input_path)
    else:
        from .process_dataset import LandmarkProcessing

        processor = LandmarkProcessing(model_path=str(hand_model_path))
        with processor.landmarker:
            pose = processor.img2landmark(str(input_path))
        if pose is None:
            raise ValueError(f"Keine gültige Handpose erkannt: {input_path}")

    return model.encode(pose)


def main():
    parser = argparse.ArgumentParser(description="Kodiert ein Handbild oder eine vorbereitete .pt-Pose.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    args = parser.parse_args()

    try:
        model = load_autoencoder(args.checkpoint)
        latent = encode_input(model, args.input_path, args.hand_model)
        print(json.dumps(latent.tolist(), allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
