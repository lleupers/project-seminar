from torch import nn

class HandAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 63),
        )

    def encode(self, poses):
        flat_poses = poses.flatten(start_dim=-2)
        return self.encoder(flat_poses)

    def decode(self, latents):
        flat_poses = self.decoder(latents)

        return flat_poses.unflatten(dim=-1, sizes=(21, 3))

    def forward(self, poses): 
        latents = self.encode(poses)
        reconstructed_poses = self.decode(latents)

        return reconstructed_poses
