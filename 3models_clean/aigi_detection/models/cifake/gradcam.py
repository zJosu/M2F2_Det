"""
Grad-CAM visualization for CIFAKENet.

Generates class-activation heatmaps highlighting regions that
contribute most to the real/fake classification decision.
Uses the pytorch-grad-cam library.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class GradCAMGenerator:
    """Generate and save Grad-CAM heatmaps for CIFAKENet.

    Args:
        model: Trained CIFAKENet instance.
        target_layer: Conv layer to hook for Grad-CAM.
            If None, uses ``model.get_gradcam_target_layer()``.
        device: Compute device.
    """

    # ImageNet de-normalisation for visualization
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ):
        self.model = model.eval().to(device)
        self.device = device

        if target_layer is None:
            target_layer = model.get_gradcam_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalised image tensor back to [0,1] for display.

        Args:
            tensor: Image tensor of shape (C, H, W).

        Returns:
            NumPy array of shape (H, W, C) in [0, 1].
        """
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.STD + self.MEAN
        return np.clip(img, 0, 1).astype(np.float32)

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> tuple:
        """Generate a Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Normalised image tensor (C, H, W).
            target_class: Class to explain (1 = FAKE by default).

        Returns:
            Tuple of (grayscale_cam, overlay_image) where grayscale_cam
            is shape (H, W) and overlay is shape (H, W, 3).
        """
        input_tensor = image_tensor.unsqueeze(0).to(self.device)
        targets = [BinaryClassifierOutputTarget(target_class)]

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # (H, W)

        rgb_image = self._denormalize(image_tensor)
        overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        return grayscale_cam, overlay

    def save_heatmaps(
        self,
        dataloader: DataLoader,
        output_dir: str,
        n_samples: int = 50,
    ) -> list:
        """Generate and save Grad-CAM heatmaps for a batch of images.

        Args:
            dataloader: DataLoader yielding (image, label, path) tuples.
            output_dir: Directory to save heatmap images.
            n_samples: Maximum number of heatmaps to generate.

        Returns:
            List of saved file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        count = 0

        for batch in tqdm(dataloader, desc="Generating Grad-CAM heatmaps"):
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [f"image_{count + i}" for i in range(len(images))]

            for i in range(len(images)):
                if count >= n_samples:
                    return saved_paths

                cam_map, overlay = self.generate_heatmap(
                    images[i], target_class=int(labels[i])
                )

                # Create side-by-side figure
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Original image
                rgb = self._denormalize(images[i])
                axes[0].imshow(rgb)
                axes[0].set_title("Original")
                axes[0].axis("off")

                # Grad-CAM heatmap
                axes[1].imshow(cam_map, cmap="jet")
                axes[1].set_title("Grad-CAM")
                axes[1].axis("off")

                # Overlay
                axes[2].imshow(overlay)
                label_str = "FAKE" if labels[i] == 1 else "REAL"
                axes[2].set_title(f"Overlay ({label_str})")
                axes[2].axis("off")

                fig.tight_layout()
                save_path = out / f"gradcam_{count:04d}_{label_str}.png"
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                saved_paths.append(str(save_path))
                count += 1

        return saved_paths
