import torch
import rasterio
import numpy as np
from torch.nn import functional as F
from pathlib import Path
from terratorch import BACKBONE_REGISTRY, DECODER_REGISTRY
import torch.nn as nn
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.models.backbones.terramind.model.terramind import TerraMind
from terratorch.models.decoders.identity_decoder import IdentityDecoder
from terratorch.tasks.classification_tasks import ClassificationTask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10

terramind_backbone = BACKBONE_REGISTRY.build(
    'terramind_v1_large',
    pretrained=True,
    modalities=['S2L1C'],
    bands={'S2L1C': ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07']}
)

terramind_backbone.to(device)

model_args = {
        # Terramind backbone
        "backbone": terramind_backbone,
        "backbone_pretrained": True,
        "backbone_modalities": ['S2L1C'],
        "backbone_bands": {'S2L1C': ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07']},
        # Identity decoder (no upsampling, just pass through features)
        "decoder": "IdentityDecoder",
        # Necks adjusted for Tiny (6 layers)
        "necks": [
            {
                "name": "SelectIndices",
                # Tiny has 6 layers (indices 0-5). 
                # We select a subset for the pyramid, e.g., layers 1, 3, 4, 5.
                "indices": [1, 3, 4, 5] 
            },
            {
                "name": "ReshapeTokensToImage",
                "remove_cls_token": False
            },
            {
                "name": "LearnedInterpolateToPyramidal"
            }
        ],
        # Classification head parameters
        "num_classes": NUM_CLASSES,
    }

model = ClassificationTask(
    model_factory="EncoderDecoderFactory",
    model_args=model_args,
)

model.to(device)
model.eval()

# Validate model setup
def validate_model_setup():
    """Validate that the ClassificationTask is properly configured."""
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 7, 64, 64).to(device)
            model_output = model.forward(dummy_input)
            
            # ClassificationTask returns a ModelOutput object with .output attribute
            logits = model_output.output
            print(f"✓ ClassificationTask returns ModelOutput with logits shape: {logits.shape}")
            
            # Verify we can compute probabilities
            if logits.dim() >= 2:
                probs = F.softmax(logits.view(logits.shape[0], -1), dim=1)
                print(f"✓ Successfully computed probabilities with shape: {probs.shape}")
            
            return True
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Validate on startup
print("Validating ClassificationTask setup...")
if not validate_model_setup():
    print("Warning: Model validation failed. Predictions may not work correctly.")

def load_tif_for_terratorch(path: str | Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)   # (C, H, W)
        # Handle the simulated panchromatic band if present (band at index 3)
        if img.shape[0] == 8:
            img = img[[0, 1, 2, 4, 5, 6, 7], :, :]  # Keep only the 7 spectral bands
        elif img.shape[0] != 7:
            raise ValueError(f"Expected 7 or 8 bands, got {img.shape[0]}")
        
    tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
    return tensor.to(device)

def extract_embedding(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Extract embeddings from encoder intermediate layers.
    Returns a list of embeddings from different layers of the encoder.
    """
    with torch.no_grad():
        # Get encoder from the ClassificationTask
        # ClassificationTask.model contains the encoder-decoder model
        if hasattr(model, 'model'):
            encoder = model.model.encoder if hasattr(model.model, 'encoder') else terramind_backbone
        else:
            encoder = terramind_backbone
        
        # Process through encoder layers to get intermediate features
        embeddings = []
        x = tensor
        
        # Extract features from encoder
        if hasattr(encoder, 'forward_features'):
            # Some encoders expose forward_features method
            x = encoder.forward_features(x)
        else:
            # Process through encoder normally
            x = encoder(x)
        
        # If output is a list of features (multi-scale), use them
        if isinstance(x, (list, tuple)):
            embeddings = list(x)
        else:
            # Single output, create a list with one embedding
            embeddings = [x]
        
        # Normalize embeddings to consistent shape
        normalized_embeddings = []
        for feat in embeddings:
            if feat.dim() == 4:
                # (B, C, H, W) -> (B, C)
                feat = feat.mean(dim=[2, 3])
            elif feat.dim() == 3:
                # (B, D, L) -> (B, D)
                feat = feat.mean(dim=[2])
            elif feat.dim() > 2:
                feat = feat.view(feat.shape[0], -1)
            
            normalized_embeddings.append(feat)
    
    return normalized_embeddings

def get_class_prediction(tif_path: str | Path) -> dict:
    """
    Get class prediction for a TIFF file using the ClassificationTask.
    Returns predicted class, probability, and per-class logits.
    """
    tensor = load_tif_for_terratorch(tif_path)
    
    with torch.no_grad():
        # Forward pass through the ClassificationTask
        # ClassificationTask returns a ModelOutput object
        model_output = model.forward(tensor)
        
        # Access the logits from the ModelOutput.output attribute
        logits = model_output.output
        
        # Ensure logits are 2D: (B, NUM_CLASSES)
        if logits.dim() == 4:
            # Pixel-wise classification - pool to image-level
            logits = logits.mean(dim=[2, 3])
        elif logits.dim() == 3:
            # Sequence output - average
            logits = logits.mean(dim=[2])
        elif logits.dim() == 1:
            # Single sample output - add batch dimension
            logits = logits.unsqueeze(0)
        
        probabilities = F.softmax(logits, dim=1)  # (B, NUM_CLASSES)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()
        
    return {
        "predicted_class": predicted_class,
        "predicted_probability": predicted_prob,
        "logits": logits[0].cpu().numpy().tolist(),
        "probabilities": probabilities[0].cpu().numpy().tolist(),
        "top_3_classes": sorted(
            [(i, float(p)) for i, p in enumerate(probabilities[0].cpu().numpy())],
            key=lambda x: x[1],
            reverse=True
        )[:3]
    }

def compare_class_predictions(file1_tif: str | Path, file2_tif: str | Path) -> dict:
    """
    Compare class predictions between two TIFF files.
    Assess class flip and probability change.
    """
    pred1 = get_class_prediction(file1_tif)
    pred2 = get_class_prediction(file2_tif)
    
    class_flip = pred1["predicted_class"] != pred2["predicted_class"]
    prob_change = pred2["predicted_probability"] - pred1["predicted_probability"]
    
    # Compute prediction stability score (higher = more stable)
    # Based on how consistent the top-3 predictions are
    top3_overlap = len(set([c for c, _ in pred1["top_3_classes"]]) & 
                       set([c for c, _ in pred2["top_3_classes"]]))
    
    return {
        "file1": str(file1_tif),
        "file2": str(file2_tif),
        "pred1_class": pred1["predicted_class"],
        "pred1_probability": pred1["predicted_probability"],
        "pred2_class": pred2["predicted_class"],
        "pred2_probability": pred2["predicted_probability"],
        "class_flip": class_flip,
        "probability_change": prob_change,
        "top3_consistency": top3_overlap / 3,  # 0-1 score
        "pred1_top3": pred1["top_3_classes"],
        "pred2_top3": pred2["top_3_classes"],
    }

def extract_spectral_statistics(tif_path: str | Path) -> dict:
    """
    Extract spectral statistics for each band in a TIFF file.
    Returns mean, std, min, max, median for all 7 spectral bands.
    """
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)  # (C, H, W)
        
        # Handle the simulated panchromatic band if present (band at index 3)
        if img.shape[0] == 8:
            img = img[[0, 1, 2, 4, 5, 6, 7], :, :]  # Keep only the 7 spectral bands
        elif img.shape[0] != 7:
            raise ValueError(f"Expected 7 or 8 bands, got {img.shape[0]}")
    
    band_names = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07']
    stats = {}
    
    for band_idx, band_name in enumerate(band_names):
        band_data = img[band_idx, :, :]
        # Filter out NoData values
        valid_data = band_data[band_data > 0]
        
        if len(valid_data) > 0:
            stats[band_name] = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "median": float(np.median(valid_data)),
                "percentile_25": float(np.percentile(valid_data, 25)),
                "percentile_75": float(np.percentile(valid_data, 75)),
            }
        else:
            stats[band_name] = {k: 0.0 for k in ["mean", "std", "min", "max", "median", "percentile_25", "percentile_75"]}
    
    return stats

def compare_spectral_signature(file1_tif: str | Path, file2_tif: str | Path) -> dict:
    """
    Compare spectral signatures between two TIFF files.
    Returns detailed analysis of band-by-band differences and drift metrics.
    """
    stats1 = extract_spectral_statistics(file1_tif)
    stats2 = extract_spectral_statistics(file2_tif)
    
    band_names = list(stats1.keys())
    drift_analysis = {
        "file1": str(file1_tif),
        "file2": str(file2_tif),
        "bands": {}
    }
    
    all_mean_diffs = []
    all_std_changes = []
    
    for band_name in band_names:
        s1 = stats1[band_name]
        s2 = stats2[band_name]
        
        # Calculate differences
        mean_diff = s2["mean"] - s1["mean"]
        mean_diff_pct = (mean_diff / s1["mean"]) * 100 if s1["mean"] != 0 else 0
        std_change = s2["std"] - s1["std"]
        std_change_pct = (std_change / s1["std"]) * 100 if s1["std"] != 0 else 0
        
        all_mean_diffs.append(abs(mean_diff))
        all_std_changes.append(abs(std_change))
        
        drift_analysis["bands"][band_name] = {
            "file1_mean": s1["mean"],
            "file2_mean": s2["mean"],
            "mean_difference": mean_diff,
            "mean_diff_percent": mean_diff_pct,
            "file1_std": s1["std"],
            "file2_std": s2["std"],
            "std_change": std_change,
            "std_change_percent": std_change_pct,
            "range_change": (s2["max"] - s2["min"]) - (s1["max"] - s1["min"]),
        }
    
    # Overall drift metrics
    drift_analysis["summary"] = {
        "avg_mean_difference": float(np.mean(all_mean_diffs)),
        "max_mean_difference": float(np.max(all_mean_diffs)),
        "avg_std_change": float(np.mean(all_std_changes)),
        "max_std_change": float(np.max(all_std_changes)),
    }
    
    return drift_analysis

def compare_stability(clean_tif: str | Path, simulated_tif: str | Path) -> dict:
    """
    Compare embedding stability between two TIFF files using the encoder.
    """
    clean_input = load_tif_for_terratorch(clean_tif)
    simu_input  = load_tif_for_terratorch(simulated_tif)
    
    # Pixel-level difference
    pixel_diff = torch.abs(clean_input - simu_input).max().item()
    
    # Extract embeddings from both inputs
    feat_clean_layers = extract_embedding(clean_input)
    feat_simu_layers  = extract_embedding(simu_input)

    # Compute similarity metrics for each layer
    cosine_sims = []
    mse_errors = []
    mae_errors = []
    
    # Handle cases where the number of layers might differ
    min_layers = min(len(feat_clean_layers), len(feat_simu_layers))
    
    for i in range(min_layers):
        feat_clean = feat_clean_layers[i]
        feat_simu = feat_simu_layers[i]
        
        # Ensure same shape for comparison
        if feat_clean.shape != feat_simu.shape:
            continue
        
        cosine_sims.append(F.cosine_similarity(feat_clean, feat_simu, dim=1).mean().item())
        mse_errors.append(F.mse_loss(feat_clean, feat_simu).item())
        mae_errors.append(F.l1_loss(feat_clean, feat_simu).item())
    
    # Return averaged metrics
    return {
        "cosine_similarity": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
        "mse_error": sum(mse_errors) / len(mse_errors) if mse_errors else 0.0,
        "mae_error": sum(mae_errors) / len(mae_errors) if mae_errors else 0.0,
        "pixel_max_diff": pixel_diff,
        "layer_cosine_similarities": cosine_sims,
        "layer_mse_errors": mse_errors,
        "layer_mae_errors": mae_errors
    }

def analyze_drift_comprehensive(file1_tif: str | Path, file2_tif: str | Path) -> dict:
    """
    Comprehensive drift analysis combining spectral signature, embedding-based metrics,
    and classification predictions. Provides a holistic view of data drift between two TIFF files.
    """
    spectral_drift = compare_spectral_signature(file1_tif, file2_tif)
    embedding_drift = compare_stability(file1_tif, file2_tif)
    class_drift = compare_class_predictions(file1_tif, file2_tif)
    
    return {
        "spectral_drift": spectral_drift,
        "embedding_drift": embedding_drift,
        "class_drift": class_drift,
        "file_pair": {
            "file1": str(file1_tif),
            "file2": str(file2_tif)
        }
    }

def print_drift_report(drift_analysis: dict, verbose: bool = False) -> None:
    """
    Pretty-print a comprehensive drift analysis report including classification predictions.
    """
    print(f"\n{'='*80}")
    print(f"SPECTRAL DRIFT ANALYSIS")
    print(f"{'='*80}")
    print(f"File 1: {Path(drift_analysis['spectral_drift']['file1']).name}")
    print(f"File 2: {Path(drift_analysis['spectral_drift']['file2']).name}")
    print()
    
    summary = drift_analysis["spectral_drift"]["summary"]
    print(f"Average mean difference across bands: {summary['avg_mean_difference']:.4f}")
    print(f"Maximum mean difference (worst band): {summary['max_mean_difference']:.4f}")
    print(f"Average std change across bands: {summary['avg_std_change']:.4f}")
    print(f"Maximum std change (worst band): {summary['max_std_change']:.4f}")
    
    if verbose:
        print(f"\nPer-band analysis:")
        for band_name, metrics in drift_analysis["spectral_drift"]["bands"].items():
            print(f"\n  {band_name}:")
            print(f"    Mean: {metrics['file1_mean']:.4f} → {metrics['file2_mean']:.4f} (Δ {metrics['mean_difference']:+.4f}, {metrics['mean_diff_percent']:+.2f}%)")
            print(f"    Std:  {metrics['file1_std']:.4f} → {metrics['file2_std']:.4f} (Δ {metrics['std_change']:+.4f}, {metrics['std_change_percent']:+.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"EMBEDDING-BASED DRIFT (TerraMind Encoder)")
    print(f"{'='*80}")
    emb = drift_analysis["embedding_drift"]
    print(f"Cosine Similarity (avg across encoder layers): {emb['cosine_similarity']:.4f}")
    print(f"MSE Error: {emb['mse_error']:.6f}")
    print(f"MAE Error: {emb['mae_error']:.6f}")
    print(f"Pixel-level max difference: {emb['pixel_max_diff']:.6f}")
    
    if verbose:
        print(f"\nPer-layer cosine similarities:")
        for i, sim in enumerate(emb['layer_cosine_similarities'], 1):
            print(f"  Layer {i:2d}: {sim:.4f}")
    
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION-BASED DRIFT (Class Flip Assessment)")
    print(f"{'='*80}")
    clf = drift_analysis["class_drift"]
    
    class_flip_indicator = "⚠️  CLASS FLIP DETECTED!" if clf["class_flip"] else "✓ No class flip"
    print(f"{class_flip_indicator}")
    print(f"\nFile 1 Prediction:")
    print(f"  Predicted class: {clf['pred1_class']} (confidence: {clf['pred1_probability']:.4f})")
    print(f"  Top 3 classes: {clf['pred1_top3']}")
    
    print(f"\nFile 2 Prediction:")
    print(f"  Predicted class: {clf['pred2_class']} (confidence: {clf['pred2_probability']:.4f})")
    print(f"  Top 3 classes: {clf['pred2_top3']}")
    
    print(f"\nPrediction Stability:")
    print(f"  Probability change: {clf['probability_change']:+.4f}")
    print(f"  Top-3 consistency: {clf['top3_consistency']:.2%}")
    
    print(f"{'='*80}\n")

def compare_directory_comprehensive(
    dir1: Path,
    dir2: Path,
    file1_pattern: str = "BANDS_RES-GRID",
    file2_pattern: str = "PHISAT2-BANDS-GRID",
    suffix: str = ".tiff"
) -> list[dict]:
    """
    Comprehensive analysis of all TIFF file pairs in two directories.
    """
    results = []
    for file1_path in sorted(dir1.glob(f"*{suffix}")):
        sim_name = file1_path.name.replace(file1_pattern, file2_pattern)
        file2_path = dir2 / sim_name
        
        if not file2_path.exists():
            print(f"[WARN] No counterpart for {file1_path.name}, skipping.")
            continue
        
        analysis = analyze_drift_comprehensive(file1_path, file2_path)
        results.append(analysis)
        
    return results

def analyze_class_flips(results: list[dict]) -> dict:
    """
    Analyze class flip statistics across multiple file pairs.
    """
    class_flips = [r for r in results if r["class_drift"]["class_flip"]]
    probabilities_changes = [r["class_drift"]["probability_change"] for r in results]
    top3_consistencies = [r["class_drift"]["top3_consistency"] for r in results]
    
    return {
        "total_pairs": len(results),
        "class_flips_count": len(class_flips),
        "class_flip_rate": len(class_flips) / len(results) if results else 0,
        "avg_probability_change": float(np.mean(probabilities_changes)) if probabilities_changes else 0,
        "max_probability_change": float(np.max(np.abs(probabilities_changes))) if probabilities_changes else 0,
        "avg_top3_consistency": float(np.mean(top3_consistencies)) if top3_consistencies else 0,
        "flipped_pairs": [
            {
                "file1": f["class_drift"]["file1"],
                "file2": f["class_drift"]["file2"],
                "class_from": f["class_drift"]["pred1_class"],
                "class_to": f["class_drift"]["pred2_class"],
                "prob_change": f["class_drift"]["probability_change"]
            }
            for f in class_flips
        ]
    }

def compare_directory(
    raw_dir: Path,
    simulated_dir: Path,
    suffix: str = ".tiff"
) -> list[dict]:
    results = []
    for raw_path in sorted(raw_dir.glob(f"*{suffix}")):
        # Naming convention: BANDS-GRID <-> PHISAT2-BANDS-GRID
        sim_name = raw_path.name.replace("BANDS_RES-GRID", "PHISAT2-BANDS-GRID")
        sim_path = simulated_dir / sim_name
    
        if not sim_path.exists():
            print(f"[WARN] No simulated counterpart for {raw_path.name}, skipping.")
            continue

        stats = compare_stability(raw_path, sim_path)
        stats["file"] = raw_path.name
        results.append(stats)
        
    return results

if __name__ == "__main__":
    # Example 1: Analyze drift for a single file pair with classification
    print("\n" + "="*80)
    print("EXAMPLE 1: Single file pair drift analysis with classification")
    print("="*80)
    
    raw_file = Path("tiff_folder/raw_tiff_update/642220-5068670_32631_BANDS_RES-GRID_0_2025-07-15T10-48-25_000.tiff")
    sim_file = Path("tiff_folder/simulated_tiff/642220-5068670_32631_PHISAT2-BANDS-GRID_0_2025-07-15T10-48-25_000.tiff")
    
    # Check if files actually exist
    raw_tiff = raw_file.with_suffix(".tiff")
    sim_tiff = sim_file.with_suffix(".tiff")
    
    if raw_tiff.exists() and sim_tiff.exists():
        drift = analyze_drift_comprehensive(raw_tiff, sim_tiff)
        print_drift_report(drift, verbose=True)
    
    # Example 2: Batch comparison with class flip detection
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch comparison with class flip detection")
    print("="*80)
    
    results = compare_directory(
        raw_dir=Path("tiff_folder/simulated_custom_tiff"),
        simulated_dir=Path("tiff_folder/simulated_custom_l2_tiff"),
    )
    print(f"\nProcessed {len(results)} image pairs.")
    
    # Average metrics across all pairs
    if results:
        avg_cos_sim = sum(r["cosine_similarity"] for r in results) / len(results)
        avg_mse = sum(r["mse_error"] for r in results) / len(results)
        avg_mae = sum(r["mae_error"] for r in results) / len(results)
        avg_pixel_diff = sum(r["pixel_max_diff"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"Average pixel-level max difference: {avg_pixel_diff:.6f}")
        print(f"Average cosine similarity (across all 12 layers): {avg_cos_sim:.4f}")
        print(f"Average MSE error: {avg_mse:.6f}")
        print(f"Average MAE error: {avg_mae:.6f}")
        print(f"{'='*60}")
        
        # Per-layer statistics
        if results and results[0]["layer_cosine_similarities"]:
            num_layers = len(results[0]["layer_cosine_similarities"])
            print(f"\nPer-layer cosine similarity statistics ({num_layers} layers):")
            for layer_idx in range(num_layers):
                layer_cos_sims = [r["layer_cosine_similarities"][layer_idx] for r in results if layer_idx < len(r["layer_cosine_similarities"])]
                if layer_cos_sims:
                    avg_layer_cos_sim = sum(layer_cos_sims) / len(layer_cos_sims)
                    print(f"  Layer {layer_idx+1:2d}: {avg_layer_cos_sim:.4f}")
    
    # Example 3: Classification-based drift analysis
    print("\n" + "="*80)
    print("EXAMPLE 3: Classification-based drift analysis (class flips)")
    print("="*80)
    
    comp_results = compare_directory_comprehensive(
        dir1=Path("tiff_folder/raw_tiff_update"),
        dir2=Path("tiff_folder/simulated_custom_l2_tiff"),
        file1_pattern="BANDS_RES-GRID",
        file2_pattern="PHISAT2-BANDS-GRID",
        suffix=".tiff"
    )
    
    if comp_results:
        class_flip_analysis = analyze_class_flips(comp_results)
        print(f"\nClass Flip Analysis:")
        print(f"  Total pairs analyzed: {class_flip_analysis['total_pairs']}")
        print(f"  Class flips detected: {class_flip_analysis['class_flips_count']}")
        print(f"  Class flip rate: {class_flip_analysis['class_flip_rate']:.2%}")
        print(f"  Average probability change: {class_flip_analysis['avg_probability_change']:+.4f}")
        print(f"  Max probability change: {class_flip_analysis['max_probability_change']:.4f}")
        print(f"  Average top-3 consistency: {class_flip_analysis['avg_top3_consistency']:.2%}")
        
        if class_flip_analysis["flipped_pairs"]:
            print(f"\n  Flipped predictions:")
            for flip in class_flip_analysis["flipped_pairs"]:
                print(f"    {Path(flip['file1']).name}")
                print(f"      Class {flip['class_from']} → {flip['class_to']} (Δ prob: {flip['prob_change']:+.4f})")