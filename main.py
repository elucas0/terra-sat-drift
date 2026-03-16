import torch
import rasterio
import numpy as np
from torch.nn import functional as F
from pathlib import Path
from terratorch import BACKBONE_REGISTRY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = BACKBONE_REGISTRY.build(
    'terramind_v1_tiny',
    pretrained=True,
    modalities=['S2L1C'],
    bands={'S2L1C': ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07']}
)
backbone.to(device)
backbone.eval()

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
    TerraMind backbone returns a LIST of multi-scale feature maps,
    one from each of the 12 transformer layers. We process each layer to get embeddings.
    Returns a list of embeddings, one per layer.
    """
    with torch.no_grad():
        feature_maps = backbone(tensor)  # List of tensors from each transformer layer

    embeddings = []
    for feat in feature_maps:
        # Flatten to 1D embedding by averaging over all spatial dimensions
        if feat.dim() == 4:
            # (B, C, H, W) -> (B, C)
            feat = feat.mean(dim=[2, 3])
        elif feat.dim() == 3:
            # (B, D, L) -> (B, D) - handle flattened spatial dims
            feat = feat.mean(dim=[2])
        else:
            # For any other shape, flatten everything except batch dimension
            feat = feat.view(feat.shape[0], -1).mean(dim=1, keepdim=True)
        embeddings.append(feat)
    
    return embeddings

def compare_stability(clean_tif: str | Path, simulated_tif: str | Path) -> dict:
    clean_input = load_tif_for_terratorch(clean_tif)
    simu_input  = load_tif_for_terratorch(simulated_tif)
    
    # Debug: Check if pixel values differ
    pixel_diff = torch.abs(clean_input - simu_input).max().item()
    
    feat_clean_layers = extract_embedding(clean_input)  # List of embeddings (one per layer)
    feat_simu_layers  = extract_embedding(simu_input)   # List of embeddings (one per layer)

    # Compute similarity metrics for each layer and average across all 12 layers
    cosine_sims = []
    mse_errors = []
    mae_errors = []
    
    for feat_clean, feat_simu in zip(feat_clean_layers, feat_simu_layers):
        cosine_sims.append(F.cosine_similarity(feat_clean, feat_simu, dim=1).item())
        mse_errors.append(F.mse_loss(feat_clean, feat_simu).item())
        mae_errors.append(F.l1_loss(feat_clean, feat_simu).item())
    
    return {
        "cosine_similarity": sum(cosine_sims) / len(cosine_sims),
        "mse_error": sum(mse_errors) / len(mse_errors),
        "mae_error": sum(mae_errors) / len(mae_errors),
        "pixel_max_diff": pixel_diff,
        "layer_cosine_similarities": cosine_sims,
        "layer_mse_errors": mse_errors,
        "layer_mae_errors": mae_errors
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
    # Batch comparison of all TIFF file pairs
    results = compare_directory(
        raw_dir=Path("tiff_folder/raw_tiff_update"),
        simulated_dir=Path("tiff_folder/simulated_custom_tiff"),
    )
    print(f"\nProcessed {len(results)} image pairs.")
    
    # Average cosine similarity across all pairs
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
        num_layers = len(results[0]["layer_cosine_similarities"])
        print(f"\nPer-layer cosine similarity statistics ({num_layers} layers):")
        for layer_idx in range(num_layers):
            layer_cos_sims = [r["layer_cosine_similarities"][layer_idx] for r in results]
            avg_layer_cos_sim = sum(layer_cos_sims) / len(layer_cos_sims)
            print(f"  Layer {layer_idx+1:2d}: {avg_layer_cos_sim:.4f}")