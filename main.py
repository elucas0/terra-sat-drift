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
    modalities=['S2L1C']
)
backbone.to(device)
backbone.eval()

# ── Data loading ──────────────────────────────────────────────────────────────
def load_tif_for_terratorch(path: str | Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)   # (C, H, W)
    img /= 10000.0                            # DN → reflectance [0, 1]
    tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
    return tensor.to(device)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_embedding(tensor: torch.Tensor) -> torch.Tensor:
    """
    FIX 4: TerraMind backbone returns a LIST of multi-scale feature maps,
    not a single tensor.  We take the last (most semantic) map and apply
    Global Average Pooling to get a compact 1-D embedding → (1, C).
    """
    with torch.no_grad():
        feature_maps = backbone(tensor)   # list[Tensor], e.g. 4 scales

    feat = feature_maps[-1]              # highest-level feature map

    # GAP over spatial dims if the map is still (1, C, H, W)
    if feat.dim() == 4:
        feat = feat.mean(dim=[2, 3])     # -> (1, C)

    return feat                          # (1, C)

# ── Stability comparison ──────────────────────────────────────────────────────
def compare_stability(clean_tif: str | Path, simulated_tif: str | Path) -> dict:
    clean_input = load_tif_for_terratorch(clean_tif)
    simu_input  = load_tif_for_terratorch(simulated_tif)

    feat_clean = extract_embedding(clean_input)   # (1, C)
    feat_simu  = extract_embedding(simu_input)    # (1, C)

    # FIX 5: cosine_similarity on dim=1 compares the two C-dim vectors correctly.
    # The old view(-1)+dim=0 collapsed everything into a single scalar dot product,
    # which is mathematically wrong for comparing embedding vectors.
    cosine_sim = F.cosine_similarity(feat_clean, feat_simu, dim=1).item()
    mse        = F.mse_loss(feat_clean, feat_simu).item()
    mae        = F.l1_loss(feat_clean, feat_simu).item()

    return {
        "cosine_similarity": cosine_sim,   # 1.0 = identical representations
        "mse_error":         mse,
        "mae_error":         mae,
    }

# ── Batch comparison over matched directory pairs ─────────────────────────────
def compare_directory(
    clean_dir: Path,
    simulated_dir: Path,
    suffix: str = ".tiff"
) -> list[dict]:
    results = []
    for clean_path in sorted(clean_dir.glob(f"*{suffix}")):
        # Assumes naming convention: clean_patch_001.tiff ↔ simulated_patch_001.tiff
        sim_name = clean_path.name.replace("clean_", "simulated_")
        sim_path = simulated_dir / sim_name

        if not sim_path.exists():
            print(f"[WARN] No simulated counterpart for {clean_path.name}, skipping.")
            continue

        stats = compare_stability(clean_path, sim_path)
        stats["file"] = clean_path.name
        results.append(stats)

        print(
            f"{clean_path.name}  |  "
            f"cos_sim={stats['cosine_similarity']:.4f}  "
            f"mse={stats['mse_error']:.6f}  "
            f"mae={stats['mae_error']:.6f}"
        )
    return results

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Single pair ---
    stats = compare_stability(
        "tiff_folder/clean_patch_001.tiff",
        "tiff_folder/simulated_patch_001.tiff"
    )
    print(f"Stability results: {stats}")

    # --- Full directory sweep (uncomment to use) ---
    # results = compare_directory(
    #     clean_dir=Path("tiff_folder/clean"),
    #     simulated_dir=Path("tiff_folder/simulated"),
    # )