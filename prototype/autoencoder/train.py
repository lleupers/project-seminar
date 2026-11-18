from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import PoseDataset
from .model import HandAutoencoder


BASE_DIR = Path(__file__).resolve().parent


def run_epoch(model, loader, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for poses in loader:
            if training:
                optimizer.zero_grad()

            reconstructed = model(poses)
            loss = loss_function(reconstructed, poses)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Der Fehler ist nicht endlich. Daten prüfen."
                )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = poses.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_autoencoder(data_dir, output_dir, epochs=100, latent_dim=16):
    """Steuert den gesamten Trainingsprozess und speichert das beste Modell."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    train_dir = data_dir / "train"
    if train_dir.is_dir() or (data_dir / "test").is_dir():
        dataset = PoseDataset(data_dir=train_dir)
    else:
        dataset = PoseDataset(data_dir=data_dir)

    if len(dataset) < 2:
        raise ValueError("Mindestens zwei Posen für Training und Validierung erforderlich.")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Training: {len(train_dataset)} Posen | Validierung: {len(val_dataset)} Posen")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = HandAutoencoder(latent_dim=latent_dim)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_files = [dataset.files[i].relative_to(data_dir).as_posix() for i in train_dataset.indices]
    val_files = [dataset.files[i].relative_to(data_dir).as_posix() for i in val_dataset.indices]

    best_val_loss = float("inf")
    model_save_path = output_dir / "autoencoder.pt"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_function, optimizer)
        val_loss = run_epoch(model, val_loader, loss_function)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_files": train_files,
                    "val_files": val_files,
                },
                model_save_path,
            )

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoche {epoch:03d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    print(f"Bestes Modell gespeichert. Validierungsfehler: {best_val_loss:.6f}")
    
    return model_save_path


if __name__ == "__main__":
    train_autoencoder(
        data_dir=BASE_DIR.parent / "data",
        output_dir=BASE_DIR / "outputs",
        epochs=100,
        latent_dim=16,
    )
