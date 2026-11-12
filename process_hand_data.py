import argparse
import glob
import os
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from utils import LandmarkProcessing

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
    args = parser.parse_args()
    processor = LandmarkProcessing("./mediapipe_models/hand_landmarker.task")

    # create train and test folders
    try:
        train_path = args.dataset_path + "/train"
        os.mkdir(train_path)
    except FileExistsError:
        print("train folder already exists.")

    try:
        test_path = args.dataset_path + "/test"
        os.mkdir(test_path)
    except FileExistsError:
        print("test folder already exists.")

    # get all image files
    glob_path = f"{args.raw_dataset_path}/**/*"
    pbar = tqdm(enumerate(glob.iglob(glob_path, recursive=True)))
    for p, path in pbar:
        # check if file is a valid image
        if (
            os.path.isfile(path)
            and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        ):
            sample = processor.img2landmark(path)
            # save sample
            image_class = path.split("/")[-2]  # get image class
            if np.random.uniform() < args.test_split:
                save_path = f"{test_path}/{p:05d}-{image_class}.pt"
            else:
                save_path = f"{train_path}/{p:05d}-{image_class}.pt"
            torch.save(sample, save_path)
