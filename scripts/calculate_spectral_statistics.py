"""
Calculate spectral statistics (mean, std, min, max) for each band
in a folder of TIFF files for normalization purposes.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import rasterio


def calculate_band_statistics(data_root: Path, sample_size: int = 100000):
    """
    Calculate per-band statistics using streaming/online algorithm.
    Processes one file at a time and frees memory immediately after.
    
    Args:
        data_root: Root directory containing TIFF files
        sample_size: Number of pixels to keep for percentile calculation
    
    Returns:
        Dictionary with statistics for each band
    """
    # Get all TIFF files
    tiff_files = sorted(Path(data_root).glob("**/*.tif")) + sorted(Path(data_root).glob("**/*.tiff"))
    print(f"Found {len(tiff_files)} TIFF files")
    
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {data_root}")
    
    # First pass: determine number of bands from first file
    with rasterio.open(tiff_files[0]) as src:
        num_bands = src.count
    
    print(f"Number of bands: {num_bands}")
    
    # Initialize Welford's algorithm accumulators (minimal memory)
    # Using Welford's online algorithm to compute mean and variance incrementally
    accumulators = {
        f"band_{i+1}": {
            "count": 0,
            "mean": 0.0,
            "M2": 0.0,  # For variance calculation
            "min": np.inf,
            "max": -np.inf,
            "samples": [],  # Keep only sample_size pixels for percentiles
        }
        for i in range(num_bands)
    }
    
    # Iterate through all TIFF files
    for tiff_file in tqdm(tiff_files, desc="Processing TIFF files"):
        try:
            with rasterio.open(tiff_file) as src:
                # Read all bands
                data = src.read()  # (bands, height, width)
                
                # Iterate through each band
                for band_idx in range(num_bands):
                    band_data = data[band_idx].flatten()
                    
                    # Filter out nodata values if present
                    if src.nodata is not None:
                        band_data = band_data[band_data != src.nodata]
                    
                    band_name = f"band_{band_idx+1}"
                    acc = accumulators[band_name]
                    
                    # Update statistics using Welford's online algorithm
                    for value in band_data:
                        acc["count"] += 1
                        delta = value - acc["mean"]
                        acc["mean"] += delta / acc["count"]
                        delta2 = value - acc["mean"]
                        acc["M2"] += delta * delta2
                        
                        # Update min/max
                        acc["min"] = min(acc["min"], value)
                        acc["max"] = max(acc["max"], value)
                        
                        # Keep reservoir sample for percentiles
                        if len(acc["samples"]) < sample_size:
                            acc["samples"].append(value)
                        else:
                            # Reservoir sampling: randomly replace
                            j = np.random.randint(0, acc["count"])
                            if j < sample_size:
                                acc["samples"][j] = value
                
                # Explicitly delete data to free memory
                del data
        
        except Exception as e:
            print(f"Warning: Could not process {tiff_file}: {e}")
            continue
    
    # Calculate final statistics from accumulators
    statistics = {}
    for band_name in sorted(accumulators.keys()):
        acc = accumulators[band_name]
        
        if acc["count"] == 0:
            print(f"Warning: No valid data for {band_name}")
            continue
        
        # Calculate variance and std from Welford's M2
        variance = acc["M2"] / (acc["count"] - 1) if acc["count"] > 1 else 0.0
        std = np.sqrt(variance)
        
        # Calculate percentiles from samples
        samples = np.array(acc["samples"])
        p2 = float(np.percentile(samples, 2)) if len(samples) > 0 else acc["mean"]
        p98 = float(np.percentile(samples, 98)) if len(samples) > 0 else acc["mean"]
        
        statistics[band_name] = {
            "mean": float(acc["mean"]),
            "std": float(std),
            "min": float(acc["min"]),
            "max": float(acc["max"]),
            "percentile_2": p2,
            "percentile_98": p98,
            "count": int(acc["count"]),
        }
    
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate spectral statistics for TIFF files (memory-efficient streaming)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/shared/home/elucas/datasets/sen1floods11_simulated/v1.1/data/flood_events/HandLabeled/S2Hand"),
        help="Root directory containing TIFF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/shared/home/elucas/datasets/sen1floods11_simulated/v1.1"),
        help="Directory to save statistics",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Number of pixels to keep for percentile calculation (lower = less memory)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Calculating statistics from {args.data_root}")
    print(f"Sample size for percentiles: {args.sample_size:,}")
    print(f"{'='*60}")
    
    try:
        stats = calculate_band_statistics(args.data_root, sample_size=args.sample_size)
        
        # Print summary
        print(f"\nSPECTRAL STATISTICS:")
        print("-" * 80)
        print(f"{'Band':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Count':<12}")
        print("-" * 80)
        for band_name, band_stats in stats.items():
            print(f"{band_name:<15} {band_stats['mean']:<12.4f} {band_stats['std']:<12.4f} "
                  f"{band_stats['min']:<12.2f} {band_stats['max']:<12.2f} {band_stats['count']:<12}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Save statistics to JSON
    output_file = args.output_dir / "spectral_statistics.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Statistics saved to {output_file}")
    
    # Create a normalization config file
    normalization_file = args.output_dir / "normalization_config.json"
    
    normalization_config = {
        "method": "standardization",
        "per_band": True,
        "statistics": {
            band_name: {
                "mean": stats[band_name]["mean"],
                "std": stats[band_name]["std"],
            }
            for band_name in stats.keys()
        }
    }
    
    with open(normalization_file, "w") as f:
        json.dump(normalization_config, f, indent=2)
    print(f"✓ Normalization config saved to {normalization_file}")


if __name__ == "__main__":
    main()
