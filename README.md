# Variational-Autoencoder
This module implements a Variational Autoencoder for image generation and
reconstruction.

Key components:
 - Encoder: Maps input images to latent distribution parameters (mu, logsigma)
 - Decoder: Reconstructs images from sampled latent vectors
 - VAE: Combines encoder and decoder with reparameterization trick
 - VAETrainer: Handles training loop with ELBO loss (reconstruction + KL divergence)

<img width="4044" height="2124" alt="W B Chart 1_24_2026, 9_16_10 PM" src="https://github.com/user-attachments/assets/582c9e5c-1a36-4471-86a9-47ebddc4a1a4" />


\
Examples of reconstructed images:

<img width="140" height="140" alt="media_images_images_999_1cd31a15a817f6d98182" src="https://github.com/user-attachments/assets/5e3f3bc7-ac84-4cff-a21b-14d36f8c2fa7" />
<img width="140" height="140" alt="media_images_images_999_33f5bc4f1206f64a1bd8" src="https://github.com/user-attachments/assets/2971bde1-d2fb-4207-9f19-663d879f8a8b" />


### Credits
I built this project while independently working through the ARENA curriculum on technical AI safety.\
Many thanks to the ARENA team for creating the program and providing the .utils files used here!
