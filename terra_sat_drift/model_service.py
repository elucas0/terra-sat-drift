"""Model loading and inference helpers for TerraMind classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from terratorch import BACKBONE_REGISTRY
from terratorch.tasks.classification_tasks import ClassificationTask


class TerraMindClassifier:
    """Wrap TerraMind + ClassificationTask setup and inference utilities."""

    BAND_NAMES = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
    BACKBONE_SIZES = {"tiny", "small", "base", "large"}
    BACKBONE_NECK_INDICES = {
        "tiny": [1, 3, 4, 5],
        "small": [1, 3, 4, 5],
        "base": [2, 5, 8, 11],
        "large": [5, 11, 17, 23],
    }

    def __init__(
        self,
        num_classes: int = 10,
        backbone_size: str = "large",
        device: torch.device | None = None,
    ) -> None:
        """Build the TerraMind-backed classification model.

        Args:
            num_classes: Number of output classes for the classification head.
            backbone_size: TerraMind model size: 'tiny', 'small', 'base', or 'large'.
            device: Optional torch device. If omitted, CUDA is used when available.

        Raises:
            ValueError: If backbone_size is not in BACKBONE_SIZES.
        """
        if backbone_size not in self.BACKBONE_SIZES:
            raise ValueError(
                f"backbone_size must be one of {self.BACKBONE_SIZES}, got {backbone_size!r}"
            )
        self.num_classes = num_classes
        self.backbone_size = backbone_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    def _build_model(self) -> ClassificationTask:
        """Create and configure the terratorch ClassificationTask model."""
        backbone_name = f"terramind_v1_{self.backbone_size}"
        neck_indices = self.BACKBONE_NECK_INDICES[self.backbone_size]

        terramind_backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=True,
            modalities=["S2L1C"],
            bands={"S2L1C": self.BAND_NAMES},
        )
        if terramind_backbone is None:
            raise RuntimeError(f"Failed to initialize TerraMind backbone: {backbone_name}")
        terramind_backbone.to(self.device)

        model_args = {
            "backbone": terramind_backbone,
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L1C"],
            "backbone_bands": {"S2L1C": self.BAND_NAMES},
            "decoder": "IdentityDecoder",
            "necks": [
                {"name": "SelectIndices", "indices": neck_indices},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "num_classes": self.num_classes,
        }

        model = ClassificationTask(model_factory="EncoderDecoderFactory", model_args=model_args)
        model.to(self.device)
        model.eval()
        return model

    def validate_setup(self) -> bool:
        """Run a lightweight model sanity check.

        Returns:
            True when model output and softmax pass complete successfully.
        """
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 7, 64, 64).to(self.device)
                model_output = self.model.forward(dummy_input)
                logits = model_output.output
                print(f"✓ ClassificationTask returns ModelOutput with logits shape: {logits.shape}")

                if logits.dim() >= 2:
                    probs = F.softmax(logits.view(logits.shape[0], -1), dim=1)
                    print(f"✓ Successfully computed probabilities with shape: {probs.shape}")
            return True
        except Exception as exc:
            print(f"✗ Model validation failed: {exc}")
            import traceback

            traceback.print_exc()
            return False

    @classmethod
    def _drop_panchromatic_if_needed(cls, img: np.ndarray) -> np.ndarray:
        """Normalize incoming arrays to exactly 7 S2 spectral bands."""
        if img.shape[0] == 8:
            return img[[0, 1, 2, 4, 5, 6, 7], :, :]
        if img.shape[0] != 7:
            raise ValueError(f"Expected 7 or 8 bands, got {img.shape[0]}")
        return img

    def load_tif_for_model(self, path: str | Path) -> torch.Tensor:
        """Load a TIFF and return model-ready tensor in BCHW format."""
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            img = self._drop_panchromatic_if_needed(img)

        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor.to(self.device)

    def extract_embeddings(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Extract normalized embeddings from encoder outputs.

        The returned embeddings are reduced to shape (B, D) for stable comparisons.
        """
        with torch.no_grad():
            encoder: Any = (
                self.model.model.encoder if hasattr(self.model.model, "encoder") else self.model
            )

            if hasattr(encoder, "forward_features"):
                encoded = encoder.forward_features(tensor)
            else:
                encoded = encoder(tensor)

            embeddings = list(encoded) if isinstance(encoded, (list, tuple)) else [encoded]
            normalized_embeddings: list[torch.Tensor] = []

            for feat in embeddings:
                if feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
                elif feat.dim() == 3:
                    feat = feat.mean(dim=[2])
                elif feat.dim() > 2:
                    feat = feat.view(feat.shape[0], -1)
                normalized_embeddings.append(feat)

        return normalized_embeddings

    def get_class_prediction(self, tif_path: str | Path) -> dict:
        """Predict class probabilities for a TIFF image."""
        tensor = self.load_tif_for_model(tif_path)

        with torch.no_grad():
            model_output = self.model.forward(tensor)
            logits = model_output.output

            if logits.dim() == 4:
                logits = logits.mean(dim=[2, 3])
            elif logits.dim() == 3:
                logits = logits.mean(dim=[2])
            elif logits.dim() == 1:
                logits = logits.unsqueeze(0)

            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            predicted_prob = probabilities[0, predicted_class].item()

        top_probs = cast(np.ndarray, probabilities[0].cpu().numpy())
        return {
            "predicted_class": predicted_class,
            "predicted_probability": predicted_prob,
            "logits": logits[0].cpu().numpy().tolist(),
            "probabilities": top_probs.tolist(),
            "top_3_classes": sorted(
                [(i, float(p)) for i, p in enumerate(top_probs)],
                key=lambda item: item[1],
                reverse=True,
            )[:3],
        }
