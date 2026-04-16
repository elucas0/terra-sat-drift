"""Inference utilities for TerraMind semantic segmentation."""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from terratorch import BACKBONE_REGISTRY
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask


class TerraMindSegmenter:
    """Wrap TerraMind + SemanticSegmentationTask for inference operations.
    
    Provides methods for loading models, performing segmentation, extracting embeddings,
    and computing metrics.
    """

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
        num_classes: int = 2,
        backbone_size: str = "base",
        device: torch.device | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        """Build the TerraMind-backed semantic segmentation model.

        Args:
            num_classes: Number of output classes for segmentation (default: 2 for binary).
            backbone_size: TerraMind model size: 'tiny', 'small', 'base', or 'large'.
            device: Optional torch device. If omitted, CUDA is used when available.
            class_names: Optional list of class names for logging.

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
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.model = self._build_model()

    def _build_model(self) -> SemanticSegmentationTask:
        """Create and configure the terratorch SemanticSegmentationTask model."""
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
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            "necks": [
                {"name": "SelectIndices", "indices": neck_indices},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "num_classes": self.num_classes,
        }

        model = SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
            class_names=self.class_names,
        )
        model.to(self.device)
        model.eval()
        return model

    def load_tif_for_model(self, path: str | Path) -> torch.Tensor:
        """Load a TIFF and return model-ready tensor in BCHW format.
        
        Args:
            path: Path to S2 TIFF file
            
        Returns:
            Tensor of shape (1, 7, H, W) on the model device
        """
        band_indices = [1, 2, 3, 7, 4, 5, 6]
        with rasterio.open(path) as src:
            s2_data = src.read().astype(np.float32)
            s2_data = s2_data[band_indices, :, :]

        tensor = torch.from_numpy(s2_data).unsqueeze(0)
        return tensor.to(self.device)

    def extract_embeddings(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Extract normalized embeddings from encoder outputs.

        The returned embeddings are reduced to shape (B, D) for stable comparisons.
        
        Args:
            tensor: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of normalized embedding tensors with shape (B, D)
        """
        with torch.no_grad():
            encoder = (
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

    def segment_image(self, tif_path: str | Path) -> dict:
        """Perform semantic segmentation on a TIFF image.
        
        Uses the TerraMind backbone with semantic segmentation decoder to produce
        a spatial segmentation map.
        
        Args:
            tif_path: Path to S2 TIFF file.
            
        Returns:
            Dictionary with:
            - 'segmentation': predicted class map (height, width)
            - 'probability_map': probability map (height, width, num_classes)
            - 'logits': raw logits (num_classes, height, width)
            - 'shape': original segmentation map shape
        """
        tensor = self.load_tif_for_model(tif_path)
        
        with torch.no_grad():
            # Get model output
            model_output = self.model.forward(tensor)
            logits = model_output.output
            
            # Ensure spatial output (batch, channels, height, width)
            if logits.dim() != 4:
                raise ValueError(
                    f"Expected 4D spatial logits, got {logits.dim()}D: {logits.shape}"
                )
                        
            # Compute probabilities via softmax
            probs = F.softmax(logits, dim=1)  # (batch, channels, height, width)
            
            # Get predicted class per pixel
            seg_map = torch.argmax(logits, dim=1)  # (batch, height, width)
        
        seg_map_np = seg_map[0].cpu().numpy().astype(np.uint8)
        logits_np = logits[0].cpu().numpy().astype(np.float32)
        prob_map_np = probs[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        
        return {
            "segmentation": seg_map_np,
            "probability_map": prob_map_np,
            "logits": logits_np,
            "shape": seg_map_np.shape,
        }

    def compute_segmentation_metrics(self, prediction: np.ndarray, 
                                      ground_truth: np.ndarray) -> dict:
        """Compute binary segmentation evaluation metrics.
        
        Args:
            prediction: Segmentation map (height, width) with class indices.
            ground_truth: Binary ground truth mask (height, width) with values 0 or 1.
            
        Returns:
            Dictionary with metrics:
            - 'iou': Intersection over Union (Jaccard index)
            - 'dice': Dice coefficient (F1 score)
            - 'accuracy': Pixel-level accuracy
            - 'precision': Precision for positive class
            - 'recall': Recall for positive class
            - 'f1_score': F1 score
            - 'confusion_matrix': {TP, FP, TN, FN}
        """
        # Convert prediction to binary (use class 1 as positive)
        if prediction.max() > 1:
            # Multi-class: convert to binary
            pred_binary = (prediction > 0).astype(np.uint8)
        else:
            pred_binary = prediction.astype(np.uint8)
        
        gt_binary = (ground_truth > 0.5).astype(np.uint8)
        
        # Confusion matrix
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        
        total = tp + fp + tn + fn
        
        # Metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        # Dice (F1 score)
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Precision & Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "iou": float(iou),
            "dice": float(dice),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
        }
