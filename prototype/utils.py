import mediapipe as mp
import torch


class LandmarkProcessing:
    def __init__(self, model_path: str):
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
        return self.image2landmark(mp_image)

    def image2landmark(self, mp_image: mp.Image) -> torch.Tensor:
        sample = torch.zeros(size=(21, 3), dtype=torch.float)
        result = self.landmarker.detect(mp_image)  # extract landmarks
        # turn landmarks into vector
        landmarks = result.hand_world_landmarks
        if len(landmarks) > 0:
            for l, landmark in enumerate(landmarks[0]):
                sample[l][0] = landmark.x
                sample[l][1] = landmark.y
                sample[l][2] = landmark.z
            norm = torch.linalg.norm(sample, dim=1).max()
            if norm > 0:
                sample = sample / norm  # normalize landmarks
        return sample
