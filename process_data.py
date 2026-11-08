import mediapipe as mp
import argparse
import glob
import os
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}


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
        "--hand_model_path",
        type=str,
        default="./mediapipe_models/hand_landmarker.task",
        help="Path to the hand pose detector",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=10,
        help="Number of times to attempt detection",
    )
    parser.add_argument(
        "--min_magnitude",
        type=float,
        default=0.1,
        help="Minimum magnitude to accept detection",
    )
    args = parser.parse_args()

    # initialize hand detectors
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.hand_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        # get all image files
        glob_path = f"{args.raw_dataset_path}/**/*"
        pbar = tqdm(enumerate(glob.iglob(glob_path, recursive=True)))
        for p, path in pbar:
            # check if file is a valid image
            if (
                os.path.isfile(path)
                and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
            ):
                for _ in range(10):
                    mp_image = mp.Image.create_from_file(path)  # load image
                    result = landmarker.detect(mp_image)  # extract landmarks

                    # turn landmarks into vector
                    sample = torch.empty(size=(21, 3), dtype=torch.float)
                    landmarks = result.hand_world_landmarks
                    for l, landmark in enumerate(landmarks):
                        sample[l] = landmark[l].x
                        sample[l] = landmark[l].y
                        sample[l] = landmark[l].z
                    # sometimes landmarks collapse to zero.
                    if (
                        torch.mean(sample) > args.min_magnitude
                    ):  # filter landmarks that collapse
                        break
                sample = (
                    sample / torch.linalg.norm(sample, dim=1).max()
                )  # normalize landmarks
                # save sample
                save_path = f"{args.dataset_path}/{p:05d}.pt"
                torch.save(sample, save_path)
