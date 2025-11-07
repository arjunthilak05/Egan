"""
Discriminator Networks

Classical neural network discriminators for QGAN training.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Standard MLP discriminator for GAN.

    Takes samples (real or generated) and outputs probability that they're real.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        use_spectral_norm: bool = False
    ):
        """
        Args:
            input_dim: Dimension of input samples
            hidden_dim: Hidden layer dimension
            n_hidden_layers: Number of hidden layers
            use_spectral_norm: Apply spectral normalization (improves stability)
        """
        super().__init__()

        layers = []

        # Input layer
        linear = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            linear = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        # Note: No sigmoid here - we'll use BCEWithLogitsLoss for numerical stability

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input samples [batch_size, input_dim]

        Returns:
            Logits [batch_size, 1] (apply sigmoid for probabilities)
        """
        return self.network(x)


class ConvDiscriminator(nn.Module):
    """
    Convolutional discriminator for image data.

    Uses strided convolutions for downsampling.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        hidden_channels: int = 64,
        use_spectral_norm: bool = False
    ):
        """
        Args:
            image_channels: Number of input channels (1 for grayscale, 3 for RGB)
            image_size: Image height/width (assumes square images)
            hidden_channels: Base number of channels in hidden layers
            use_spectral_norm: Apply spectral normalization
        """
        super().__init__()

        self.image_channels = image_channels
        self.image_size = image_size

        # Calculate feature map sizes after convolutions
        # Each conv with stride 2 halves the spatial dimensions

        def conv_block(in_channels, out_channels, use_bn=True):
            layers = []
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)

            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        # Convolutional layers
        self.conv1 = conv_block(image_channels, hidden_channels, use_bn=False)
        self.conv2 = conv_block(hidden_channels, hidden_channels * 2)
        self.conv3 = conv_block(hidden_channels * 2, hidden_channels * 4)

        # Calculate flattened size
        # After 3 conv layers with stride 2: size = image_size // 8
        final_size = image_size // 8
        flattened_size = hidden_channels * 4 * final_size * final_size

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [batch_size, channels, height, width]

        Returns:
            Logits [batch_size, 1]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies patches as real/fake.

    More efficient than standard discriminator for high-resolution images.
    """

    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 3
    ):
        super().__init__()

        def conv_block(in_channels, out_channels, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        layers = [conv_block(image_channels, hidden_channels, stride=2)]

        for i in range(1, n_layers):
            mult = 2 ** i
            layers.append(conv_block(hidden_channels * mult // 2, hidden_channels * mult))

        # Final layer
        layers.append(nn.Conv2d(hidden_channels * mult, 1, kernel_size=4, stride=1, padding=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [batch_size, channels, height, width]

        Returns:
            Patch predictions [batch_size, 1, patch_h, patch_w]
        """
        return self.network(x)


class WassersteinDiscriminator(Discriminator):
    """
    Discriminator for Wasserstein GAN (WGAN).

    Same architecture as standard discriminator but without sigmoid activation.
    Acts as a critic rather than classifier.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_hidden_layers: int = 3):
        super().__init__(input_dim, hidden_dim, n_hidden_layers, use_spectral_norm=True)
        # WGAN critics benefit from spectral normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns Wasserstein distance estimate (not a probability).
        """
        return self.network(x)


class GradientPenaltyMixin:
    """
    Mixin for computing gradient penalty in WGAN-GP.
    """

    @staticmethod
    def compute_gradient_penalty(
        discriminator: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        Args:
            discriminator: Discriminator network
            real_samples: Real data samples
            fake_samples: Generated samples
            lambda_gp: Gradient penalty weight

        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Random interpolation weight
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Discriminator output on interpolates
        d_interpolates = discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty
