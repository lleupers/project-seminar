from pathlib import Path
from urllib.request import urlopen, urlretrieve

import cv2

import mediapipe as mp
from utils import LandmarkProcessing

import torch

from autoencoder.model import HandAutoencoder
from components.kNN import load_latents, predict, train_knn
import numpy as np

from string import ascii_uppercase

def encode_clip(video_path, model, processor):
    video = cv2.VideoCapture(str(video_path))
    embeddings = []

    try:
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise RuntimeError(f"Keine Frames gefunden: {video_path}")

        positions = np.linspace(0.25, 0.75, num=9)
        frame_indices = sorted({
            round(position * (frame_count - 1))
            for position in positions
        })

        with torch.inference_mode():
            for index in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = video.read()
                if not ok:
                    raise RuntimeError(f"Frame {index} konnte nicht gelesen werden.")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose = processor.image2landmark(image)

                if pose.any():
                    embeddings.append(model.encode(pose))
    finally:
        video.release()

    if not embeddings:
        return None

    print(f"Frames mit erkannter Hand: {len(embeddings)}/{len(frame_indices)}")
    return torch.stack(embeddings).mean(dim=0)

base_url = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/I6_Gestures"
index_url = f"{base_url}/i6_data_P20_BOTH.txt"


with urlopen(index_url, timeout=30) as response: 
    index_text = response.read().decode("utf-8")

lines = index_text.splitlines()

labels_by_id = {
    str(index): letter
    for index, letter in enumerate(ascii_uppercase, start=1)
    if letter not in {"J", "Z"}
}

selected_clips = []

for line in lines:
    if line.endswith(".seq"):
        sequence_path = Path(line)
        gesture_id = sequence_path.parent.name

        if gesture_id in labels_by_id:
            selected_clips.append((gesture_id, sequence_path))

print(f"Ausgewählte Clips: {len(selected_clips)}")


gesture_id = sequence_path.parent.name
video_name = sequence_path.with_suffix(".mpg").name
video_url = f"{base_url}/video/{gesture_id}/{video_name}"

print(video_url)

destination = Path("raw_data/rwth") / gesture_id / video_name
destination.parent.mkdir(parents=True, exist_ok=True)

urlretrieve(video_url, destination)



checkpoint_path = Path("autoencoder/outputs/autoencoder.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

model = HandAutoencoder(latent_dim=checkpoint["latent_dim"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

reference_path = Path("autoencoder/outputs/train_latents.csv")
x_train, y_train, sample_ids = load_latents(reference_path)
classifier = train_knn(x_train, y_train, k=1, metric="cosine")

processor = LandmarkProcessing("mediapipe_models/hand_landmarker.task")


correct = 0 
total = 0

try: 
    for gesture_id, sequence_path in selected_clips:
        video_name = sequence_path.with_suffix(".mpg").name
        destination = Path("raw_data/rwth") / gesture_id / video_name

        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            video_url= f"{base_url}/video/{gesture_id}/{video_name}"
            urlretrieve(video_url, destination)

        embedding = encode_clip(destination, model, processor)
        expected = labels_by_id[gesture_id]
        total += 1

        if embedding is None:
            print(f"{video_name} | Soll={expected} | Keine Hand erkannt")
            continue
        prediction = predict(classifier, embedding.numpy())
    
        if prediction == expected:
            correct += 1

        print(f"Vorhersage für {video_name}: {prediction}, erwartet: {expected}")

finally: 
    processor.landmarker.close()

if total > 0:
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")


    