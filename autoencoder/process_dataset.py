import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import mediapipe as mp
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
# from utils import LandmarkProcessing  # Original: korrigierte Fassung unten

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}


class LandmarkProcessing:
    def __init__(
        self, model_path: str, min_magnitude: float = 0.0, max_attempts: int = 1
    ):
        self.min_magnitude = min_magnitude
        self.max_attempts = max_attempts
        # initialize hand detector
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def img2landmark(self, path: str) -> torch.Tensor | None:
        mp_image = mp.Image.create_from_file(path)
        for _ in range(self.max_attempts):
            result = self.landmarker.detect(mp_image)

            # Original aus utils.py: Die äußere Liste enthält Hände.
            # landmarks = result.hand_world_landmarks
            # for l, landmark in enumerate(landmarks):
            #     sample[l] = landmark[l].x
            #     sample[l] = landmark[l].y
            #     sample[l] = landmark[l].z

            if not result.hand_world_landmarks:
                continue

            # Erst die Hand auswählen, dann pro Punkt eine Zeile befüllen.
            landmarks = result.hand_world_landmarks[0]
            sample = torch.empty((21, 3), dtype=torch.float32)
            for index, point in enumerate(landmarks):
                sample[index, 0] = point.x
                sample[index, 1] = point.y
                sample[index, 2] = point.z

            # Original: Der Mittelwert vorzeichenbehafteter Koordinaten
            # kann auch bei einer gültigen Pose nahe null sein.
            # if torch.mean(sample) > self.min_magnitude:
            #     break
            # sample = sample / torch.linalg.norm(sample, dim=1).max()
            # return sample

            norm = torch.linalg.norm(sample, dim=1).max()
            if torch.isfinite(norm) and norm > max(0.0, self.min_magnitude):
                return sample / norm

        return None


class HandDataset(Dataset):
    """Dataset for hand autoencoder.
    Takes a filepath to a folder of .pt files and returns a set of 3d-vectors representing mediapipe hand landmarks

    Args:
        Dataset (str): Path to folder of .pt-files
    """

    def __init__(
        self,
        dataset_path: str = "./data",
    ):
        self.files = glob.glob(f"{dataset_path}/*.pt")

    def __len__(self) -> int:
        """Returns length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """Returns two tensors of 21 3d-vectors representing mediapipe hand landmarks for training an autoencoder

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.tensor, torch.tensor]: two tensors of 21 3d-vectors
        """
        sample_path = self.files[idx]
        return (torch.load(sample_path), torch.load(sample_path))


def process_images(images, args, train_path, test_path, progress):
    # processor = LandmarkProcessing("./mediapipe_models/hand_landmarker.task")
    processor = LandmarkProcessing(
        model_path=args.hand_model_path,
        min_magnitude=args.min_magnitude,
        max_attempts=args.num_retries,
    )
    with processor.landmarker:
        for index, path in images:
            sample = processor.img2landmark(path)
            if sample is not None:
                # image_class = path.split("/")[-2]  # Original: nur "/" als Trennzeichen
                image_class = os.path.basename(os.path.dirname(path))
                target = test_path if np.random.uniform() < args.test_split else train_path
                torch.save(sample, os.path.join(target, f"{index:05d}-{image_class}.pt"))
            progress.update()


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Path to the dataset of hand poses",
    )
    parser.add_argument(
        "--raw_dataset_path",
        type=str,
        default="./raw_data",
        help="Path to the raw dataset of hand images",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split of dataset",
    )
    parser.add_argument(
        "--hand_model_path",
        type=str,
        default="./mediapipe_models/hand_landmarker.task",
        help="Path to the hand pose detector",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=1,
        help="Number of times to attempt detection",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of images to process concurrently (one hand landmarker per worker)",
    )
    parser.add_argument(
        "--min_magnitude",
        type=float,
        # default=0.1,  # Original: bezog sich auf den Koordinatenmittelwert
        default=0.0,
        help="Minimum maximum landmark distance from the origin, in meters",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.num_retries < 1:
        parser.error("--num_retries must be at least 1")

    # os.mkdir(train_path)  # Original: setzt einen bestehenden Elternordner voraus
    # os.mkdir(test_path)
    train_path = os.path.join(args.dataset_path, "train")
    test_path = os.path.join(args.dataset_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    glob_path = f"{args.raw_dataset_path}/**/*"
    images = [
        (index, path)
        for index, path in enumerate(glob.iglob(glob_path, recursive=True))
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    ]
    if images:
        worker_count = min(args.workers, len(images))
        with tqdm(total=len(images), unit="Bild") as progress:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                jobs = [
                    executor.submit(
                        process_images, images[offset::worker_count], args,
                        train_path, test_path, progress,
                    )
                    for offset in range(worker_count)
                ]
                for job in jobs:
                    job.result()
