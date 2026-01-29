"""
Variational Autoencoder (VAE) Implementation

This module implements a Variational Autoencoder for image generation and
reconstruction. The implementation follows the original VAE paper (Kingma & Welling, 2013)
with a convolutional architecture suitable for MNIST and CelebA datasets.

Key components:
    - Encoder: Maps input images to latent distribution parameters (mu, logsigma)
    - Decoder: Reconstructs images from sampled latent vectors
    - VAE: Combines encoder and decoder with reparameterization trick
    - VAETrainer: Handles training loop with ELBO loss (reconstruction + KL divergence)

The code is organized as follows:
    1. Imports and device configuration
    2. Dataset loading utilities
    3. Network architectures (Encoder, Decoder, VAE)
    4. Training configuration and trainer
    5. Main execution
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t
import torch.nn as nn
import wandb
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


# ============================================================================
# Device Configuration
# ============================================================================

# Device selection: prefer CUDA > MPS > CPU
device = t.device(
    "cuda" if t.cuda.is_available()
    else "mps" if t.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================================
# Dataset Utilities
# ============================================================================

def get_dataset(dataset: Literal["MNIST", "CIFAR10"], train: bool = True) -> Dataset:
    """
    Purpose:
        Load and prepare a dataset with appropriate transforms for VAE training.
        Supports MNIST (grayscale, 28x28) and CIFAR10 (RGB, 32x32).

    Parameters:
     * dataset (Literal["MNIST", "CIFAR10"]) : name of dataset to load
     * train (bool) : if True, load training set; otherwise load test set

    Returns:
        PyTorch Dataset object with images normalized to [-1, 1]
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
        ])
        return datasets.MNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=True
        )

    elif dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
        return datasets.CIFAR10(
            root=data_dir,
            train=train,
            transform=transform,
            download=True
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_holdout_samples(dataset: Literal["MNIST", "CIFAR10"], num_samples: int = 10) -> Tensor:
    """
    Purpose:
        Create a fixed set of holdout images for consistent evaluation during training.
        For MNIST, selects one image of each digit 0-9.

    Parameters:
     * dataset (Literal["MNIST", "CIFAR10"]) : dataset to sample from
     * num_samples (int) : number of samples to extract

    Returns:
        Tensor of shape (num_samples, C, H, W) containing holdout images
    """
    testset = get_dataset(dataset, train=False)
    holdout_dict = {}

    for img, label in DataLoader(testset, batch_size=1, shuffle=False):
        if label.item() not in holdout_dict:
            holdout_dict[label.item()] = img.squeeze(0)
            if len(holdout_dict) >= num_samples:
                break

    # Stack into single tensor
    holdout_images = t.stack([holdout_dict[i] for i in sorted(holdout_dict.keys())])
    return holdout_images.to(device)


# ============================================================================
# Network Architectures
# ============================================================================

class Encoder(nn.Module):
    """
    Purpose:
        Encoder network that maps input images to latent distribution parameters.
        Uses convolutional layers followed by fully connected layers to output
        mean (mu) and log standard deviation (logsigma) for the latent Gaussian.
    """

    def __init__(self, input_channels: int, latent_dim: int, hidden_dim: int):
        """
        Purpose:
            Initialize encoder architecture with conv layers and fc layers.

        Parameters:
         * input_channels (int) : number of input channels (1 for MNIST, 3 for CIFAR10)
         * latent_dim (int) : dimensionality of the latent space
         * hidden_dim (int) : dimensionality of the hidden fully-connected layer
        """
        super().__init__()

        # Convolutional feature extraction
        # For 28x28 input: 28 -> 14 -> 7
        # For 32x32 input: 32 -> 16 -> 8
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
        )

        # Calculate flattened dimension (depends on input size)
        # MNIST (28x28): 64 * 7 * 7 = 3136
        # CIFAR10 (32x32): 64 * 8 * 8 = 4096
        if input_channels == 1:  # MNIST
            self.flattened_dim = 64 * 7 * 7
        else:  # CIFAR10
            self.flattened_dim = 64 * 8 * 8

        # Fully connected layers to latent parameters
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),  # Output both mu and logsigma
            Rearrange('b (n latent_dim) -> n b latent_dim', n=2),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Purpose:
            Forward pass through encoder to obtain latent distribution parameters.

        Parameters:
         * x (Tensor) : input images of shape (batch, channels, height, width)

        Returns:
            Tuple of (mu, logsigma) each with shape (batch, latent_dim)
        """
        x = self.conv_layers(x)
        output = self.fc_layers(x)
        mu = output[0]
        logsigma = output[1]
        return mu, logsigma


class Decoder(nn.Module):
    """
    Purpose:
        Decoder network that reconstructs images from latent vectors.
        Uses fully connected layers followed by transposed convolutions
        to upsample from latent space back to image space.
    """

    def __init__(self, output_channels: int, latent_dim: int, hidden_dim: int):
        """
        Purpose:
            Initialize decoder architecture with fc layers and transposed conv layers.

        Parameters:
         * output_channels (int) : number of output channels (1 for MNIST, 3 for CIFAR10)
         * latent_dim (int) : dimensionality of the latent space
         * hidden_dim (int) : dimensionality of the hidden fully-connected layer
        """
        super().__init__()

        self.output_channels = output_channels

        # Calculate starting spatial dimension
        if output_channels == 1:  # MNIST
            self.start_h, self.start_w = 7, 7
        else:  # CIFAR10
            self.start_h, self.start_w = 8, 8

        flattened_dim = 64 * self.start_h * self.start_w

        # Fully connected layers from latent to feature maps
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flattened_dim),
            nn.ReLU(),
            Rearrange('b (c h w) -> b c h w', c=64, h=self.start_h, w=self.start_w),
        )

        # Transposed convolutions for upsampling
        # 7x7 -> 14x14 -> 28x28 (MNIST)
        # 8x8 -> 16x16 -> 32x32 (CIFAR10)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # x2
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Purpose:
            Forward pass through decoder to reconstruct images from latent vectors.

        Parameters:
         * z (Tensor) : latent vectors of shape (batch, latent_dim)

        Returns:
            Reconstructed images of shape (batch, channels, height, width)
        """
        x = self.fc_layers(z)
        x = self.deconv_layers(x)
        return x


class VAE(nn.Module):
    """
    Purpose:
        Variational Autoencoder combining encoder and decoder with reparameterization trick.
        Implements the full VAE forward pass: encode -> sample latent -> decode.
    """

    def __init__(self, input_channels: int, latent_dim: int, hidden_dim: int):
        """
        Purpose:
            Initialize VAE with encoder and decoder networks.

        Parameters:
         * input_channels (int) : number of input/output channels
         * latent_dim (int) : dimensionality of the latent space
         * hidden_dim (int) : dimensionality of hidden layers in encoder/decoder
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(input_channels, latent_dim, hidden_dim)
        self.decoder = Decoder(input_channels, latent_dim, hidden_dim)

    def sample_latent(self, mu: Tensor, logsigma: Tensor) -> Tensor:
        """
        Purpose:
            Sample from latent distribution using reparameterization trick.
            z = mu + sigma * epsilon, where epsilon ~ N(0, 1)

        Parameters:
         * mu (Tensor) : mean of latent distribution, shape (batch, latent_dim)
         * logsigma (Tensor) : log std of latent distribution, shape (batch, latent_dim)

        Returns:
            Sampled latent vector z of shape (batch, latent_dim)
        """
        sigma = t.exp(logsigma)
        epsilon = t.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Purpose:
            Full forward pass through VAE: encode input, sample latent, decode.

        Parameters:
         * x (Tensor) : input images of shape (batch, channels, height, width)

        Returns:
            Tuple of (reconstruction, mu, logsigma):
             * reconstruction (Tensor) : reconstructed images, same shape as input
             * mu (Tensor) : latent distribution mean, shape (batch, latent_dim)
             * logsigma (Tensor) : latent distribution log std, shape (batch, latent_dim)
        """
        # Encode to latent parameters
        mu, logsigma = self.encoder(x)

        # Sample latent vector using reparameterization trick
        z = self.sample_latent(mu, logsigma)

        # Decode to reconstruct image
        reconstruction = self.decoder(z)

        return reconstruction, mu, logsigma

    @t.inference_mode()
    def generate(self, num_samples: int) -> Tensor:
        """
        Purpose:
            Generate new images by sampling from the prior N(0, 1) and decoding.

        Parameters:
         * num_samples (int) : number of images to generate

        Returns:
            Generated images of shape (num_samples, channels, height, width)
        """
        # Sample from standard normal prior
        z = t.randn(num_samples, self.latent_dim, device=next(self.parameters()).device)

        # Decode to images
        images = self.decoder(z)
        return images


# ============================================================================
# Training Configuration and Trainer
# ============================================================================

@dataclass
class VAEArgs:
    """
    Purpose:
        Configuration dataclass containing all hyperparameters for VAE training.
    """

    # === Architecture ===
    latent_dim: int = 10  # Dimensionality of latent space
    hidden_dim: int = 256  # Hidden layer size in encoder/decoder

    # === Dataset ===
    dataset: Literal["MNIST", "CIFAR10"] = "MNIST"  # Dataset to train on

    # === Training ===
    batch_size: int = 128  # Training batch size
    epochs: int = 10  # Number of training epochs
    lr: float = 1e-3  # Learning rate
    beta_kl: float = 1.0  # Weight for KL divergence term (β-VAE)

    # === Logging ===
    use_wandb: bool = False  # Whether to log to Weights & Biases
    wandb_project: str = "vae-mnist"  # W&B project name
    wandb_name: str | None = None  # W&B run name
    log_every_n_steps: int = 100  # Log reconstructions every N batches


class VAETrainer:
    """
    Purpose:
        Handles the full training loop for VAE, including data loading, optimization,
        loss computation, and logging.
    """

    def __init__(self, args: VAEArgs):
        """
        Purpose:
            Initialize trainer with model, data, and optimizer.

        Parameters:
         * args (VAEArgs) : training configuration
        """
        self.args = args

        # Determine number of input channels based on dataset
        self.input_channels = 1 if args.dataset == "MNIST" else 3

        # Load dataset
        self.trainset = get_dataset(args.dataset, train=True)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        # Initialize model
        self.model = VAE(
            input_channels=self.input_channels,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim
        ).to(device)

        # Initialize optimizer
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr)

        # Get holdout samples for visualization
        self.holdout_samples = get_holdout_samples(args.dataset, num_samples=10)

        # Training state
        self.step = 0

    def compute_loss(
        self,
        x: Tensor,
        reconstruction: Tensor,
        mu: Tensor,
        logsigma: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Purpose:
            Compute VAE loss (ELBO) consisting of reconstruction loss and KL divergence.
            ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
            We minimize the negative ELBO: reconstruction_loss + beta_kl * kl_divergence

        Parameters:
         * x (Tensor) : original input images
         * reconstruction (Tensor) : reconstructed images from decoder
         * mu (Tensor) : latent distribution mean from encoder
         * logsigma (Tensor) : latent distribution log std from encoder

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_divergence):
             * total_loss (Tensor) : combined loss for backpropagation
             * reconstruction_loss (Tensor) : MSE between input and reconstruction
             * kl_divergence (Tensor) : KL divergence from standard normal prior
        """
        # Reconstruction loss: MSE between input and reconstruction
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')

        # KL divergence: KL(q(z|x) || N(0, 1))
        # Closed form: -0.5 * sum(1 + 2*logsigma - mu^2 - sigma^2)
        kl_divergence = -0.5 * t.mean(
            1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp()
        )

        # Total loss with beta weighting for KL term (β-VAE)
        total_loss = reconstruction_loss + self.args.beta_kl * kl_divergence

        return total_loss, reconstruction_loss, kl_divergence

    def training_step(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Purpose:
            Perform a single training step: forward pass, loss computation,
            backpropagation, and optimizer update.

        Parameters:
         * x (Tensor) : batch of input images

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_divergence)
        """
        # Forward pass through VAE
        reconstruction, mu, logsigma = self.model(x)

        # Compute loss
        total_loss, recon_loss, kl_loss = self.compute_loss(x, reconstruction, mu, logsigma)

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, recon_loss, kl_loss

    @t.inference_mode()
    def log_reconstructions(self) -> None:
        """
        Purpose:
            Evaluate model on holdout samples and log reconstructions to wandb.
            This provides visual feedback on training progress.
        """
        self.model.eval()

        # Get reconstructions of holdout samples
        reconstruction, _, _ = self.model(self.holdout_samples)

        # Normalize to [0, 1] for display
        reconstruction = (reconstruction + 1) / 2  # From [-1, 1] to [0, 1]
        reconstruction = t.clamp(reconstruction, 0, 1)

        if self.args.use_wandb:
            # Convert to uint8 for wandb logging
            reconstruction = (reconstruction * 255).to(dtype=t.uint8)

            wandb.log({
                "reconstructions": [
                    wandb.Image(img) for img in reconstruction.cpu().numpy()
                ]
            }, step=self.step)

        self.model.train()

    def train(self) -> VAE:
        """
        Purpose:
            Execute full training loop over all epochs.

        Returns:
            Trained VAE model
        """
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model, log='all', log_freq=100)

        print(f"Training VAE on {self.args.dataset}")
        print(f"Device: {device}")
        print(f"Latent dim: {self.args.latent_dim}, Hidden dim: {self.args.hidden_dim}")
        print(f"Beta KL: {self.args.beta_kl}\n")

        for epoch in range(self.args.epochs):
            self.model.train()

            # Progress bar for current epoch
            progress_bar = tqdm(
                self.trainloader,
                desc=f"Epoch {epoch+1}/{self.args.epochs}",
                ascii=True
            )

            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(device)

                # Training step
                total_loss, recon_loss, kl_loss = self.training_step(images)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })

                # Log to wandb
                if self.args.use_wandb:
                    wandb.log({
                        'total_loss': total_loss.item(),
                        'reconstruction_loss': recon_loss.item(),
                        'kl_divergence': kl_loss.item(),
                        'epoch': epoch
                    }, step=self.step)

                # Log reconstructions periodically
                if self.step % self.args.log_every_n_steps == 0:
                    self.log_reconstructions()

                self.step += 1

        # Final reconstruction logging
        self.log_reconstructions()

        if self.args.use_wandb:
            wandb.finish()

        print("\nTraining complete!")
        return self.model


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configure training
    args = VAEArgs(
        latent_dim=10,
        hidden_dim=256,
        dataset="MNIST",
        batch_size=128,
        epochs=10,
        lr=1e-3,
        beta_kl=1.0,
        use_wandb=True,
        log_every_n_steps=100
    )

    # Train model
    trainer = VAETrainer(args)
    vae = trainer.train()

    # Save trained model
    save_path = Path(__file__).parent / "vae_model.pth"
    t.save(vae.state_dict(), save_path)
    print(f"Model saved to {save_path}")
