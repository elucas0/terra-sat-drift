#!/usr/bin/env python3
"""
Script to resample label files to 4.75m resolution.
Handles label files from Sen1Floods11 simulated dataset.
"""

import numpy as np
import rasterio
from pathlib import Path
from eolearn.features.utils import spatially_resize_image, ResizeMethod
import logging

from terra_sat_drift.data_simulation.phisat2_constants import PHISAT2_RESOLUTION, S2_RESOLUTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LABEL_DIR = Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand")
OUTPUT_DIR = Path("/shared/home/elucas/datasets/sen1floods11_simulated/v1.1/data/flood_events/HandLabeled/LabelHand")
TARGET_RESOLUTION = 4.75  # meters

# Current resolution of Sen1Floods11 labels (typically 10m for Sentinel-2 based data)
CURRENT_RESOLUTION = 10.0  # meters


def resample_label_file(input_path: Path, output_path: Path) -> bool:
    """
    Resample a single label file to new resolution.
    
    Args:
        input_path: Path to input label file
        output_path: Path to save resampled label file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with rasterio.open(input_path) as src:
            # Read label data
            data = src.read()
            profile = src.profile
            
            # Handle different data shapes
            if data.ndim == 3:
                # Single band - squeeze to 2D
                if data.shape[0] == 1:
                    data = data[0]
                else:
                    # Multiple bands - stack them
                    data = np.transpose(data, (1, 2, 0))
            
            # Calculate new size based on scale factor
            if data.ndim == 2:
                height, width = data.shape
            else:
                height, width = data.shape[0], data.shape[1]
            
            new_height = int((height * S2_RESOLUTION) / PHISAT2_RESOLUTION)
            new_width = int((width * S2_RESOLUTION) / PHISAT2_RESOLUTION)
            new_size = (new_width, new_height)
            
            logger.info(f"Resampling {input_path.name}: {height}x{width} -> {new_height}x{new_width}")
            
            # Resample using nearest neighbor for labels (to preserve class values)
            resampled = spatially_resize_image(
                data,
                new_size=new_size,
                resize_method=ResizeMethod.NEAREST,
            )
            
            new_transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': new_transform,
                'dtype': data.dtype
            })
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(resampled.astype(profile['dtype']), 1)
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Failed to resample {input_path.name}: {e}")
        return False


def main():
    """Process all label files in the input directory."""
    
    if not LABEL_DIR.exists():
        logger.error(f"Input directory not found: {LABEL_DIR}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
        
    # Find all label files
    label_files = list(LABEL_DIR.glob("*.tif"))
    
    if not label_files:
        logger.warning(f"No .tif/.tiff files found in {LABEL_DIR}")
        return
    
    logger.info(f"Found {len(label_files)} label files to process")
    
    # Process each file
    success_count = 0
    for label_file in label_files:
        output_file = OUTPUT_DIR / label_file.name
        if resample_label_file(label_file, output_file):
            success_count += 1
    
    logger.info(f"\nProcessing complete: {success_count}/{len(label_files)} files successfully resampled")
    logger.info(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
