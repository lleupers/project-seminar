import mediapipe as mp
import torch


class LandmarkProcessing:
    def __init__(
        self, model_path: str, min_magnitude: float = 0.1, max_attempts: int = 10
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

    def img2landmark(self, path: str) -> torch.tensor:
        mp_image = mp.Image.create_from_file(path)  # load image
        sample = torch.empty(size=(21, 3), dtype=torch.float)
        for _ in range(self.max_attempts):
            result = self.landmarker.detect(mp_image)  # extract landmarks
            # turn landmarks into vector
            landmarks = result.hand_world_landmarks
            for l, landmark in enumerate(landmarks):
                sample[l] = landmark[l].x
                sample[l] = landmark[l].y
                sample[l] = landmark[l].z
            # sometimes landmarks collapse to zero.
            if (
                torch.mean(sample) > self.min_magnitude
            ):  # filter landmarks that collapse
                break
        sample = sample / torch.linalg.norm(sample, dim=1).max()  # normalize landmarks
        return sample
