"""Unified loader for Sen1Floods11 drift analysis with raw, simulated, and ground truth data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import rasterio
import json


class Sen1FloodsDriftLoader:
    """Loader for Sen1Floods11 drift analysis with raw S2, simulated S2, and ground truth masks.
    
    This loader pairs raw S2 imagery from the Sen1Floods11 dataset with:
    - Simulated Φ-sat-2 versions from the simulated_sen1floods folder
    - Binary ground truth masks from the LabelHand directory
    
    Enables comprehensive drift analysis including spectral, embedding, and segmentation metrics.
    """

    def __init__(
        self,
        sen1floods_root: Path | str,
        simulated_dir: Path | str,
        split: str = "train",
        dataset_type: str = "HandLabeled",
    ):
        """Initialize the drift loader for Sen1Floods11.

        Args:
            sen1floods_root: Root path to the Sen1Floods11 dataset 
                           (e.g., datasets/sen1floods11).
            simulated_dir: Directory containing simulated Φ-sat-2 .tif files
                         (e.g., tiff_folder/simulated_sen1floods).
            split: Dataset split - one of ['train', 'valid', 'test'].
            dataset_type: Type of dataset - 'HandLabeled' or 'WeaklyLabeled'.
        """
        self.sen1floods_root = Path(sen1floods_root) / "v1.1"
        self.simulated_dir = Path(simulated_dir)
        self.split = split
        self.dataset_type = dataset_type

        # Validate split
        split_mapping = {
            "train": "train", 
            "val": "valid", 
            "valid": "valid", 
            "test": "test"
        }
        self.split_mapped = split_mapping.get(split, split)

        # Paths
        self.split_file = (
            self.sen1floods_root 
            / f"splits/flood_{dataset_type.lower()}" 
            / f"flood_{self.split_mapped}_data.csv"
        )
        self.data_root = self.sen1floods_root / f"data/flood_events/{dataset_type}"
        self.s2_dir = self.data_root / "S2Hand"
        self.label_dir = self.data_root / "LabelHand"
        self.metadata_file = self.sen1floods_root / "Sen1Floods11_Metadata.geojson"

        # Validate required paths
        self._validate_paths()
        
        # Load and pair data
        self._load_pairs()

    def _validate_paths(self) -> None:
        """Validate that required directories exist."""
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.s2_dir.exists():
            raise FileNotFoundError(f"S2 directory not found: {self.s2_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        if not self.simulated_dir.exists():
            raise FileNotFoundError(f"Simulated directory not found: {self.simulated_dir}")

    def _load_pairs(self) -> None:
        """Load split CSV and pair raw S2 with simulated and labels."""
        # Try to load metadata for location info
        try:
            import geopandas
            self.metadata = geopandas.read_file(str(self.metadata_file))
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            self.metadata = None

        # Read split file
        with open(self.split_file) as f:
            file_list = f.readlines()

        file_list = [line.rstrip().split(",") for line in file_list]

        # Build pairs: (raw_s2, simulated_s2, label)
        self.pairs: List[Dict[str, Path]] = []
        
        for s1_file, label_file in file_list:
            # Convert S1Hand filename to S2Hand
            s2_filename = s1_file.replace("S1Hand", "S2Hand")
            raw_s2_path = self.s2_dir / s2_filename
            
            # Build label path
            label_path = self.label_dir / label_file
            
            # Build simulated file path
            # Extract base name and create simulated filename
            simulated_filename = f"simulated_{s2_filename}"
            simulated_path = self._find_simulated_file(simulated_filename, s2_filename)
            
            # Only add pair if raw S2 and label exist
            if raw_s2_path.exists() and label_path.exists():
                self.pairs.append({
                    "raw_s2": raw_s2_path,
                    "simulated_s2": simulated_path,
                    "label": label_path,
                    "s2_filename": s2_filename,
                })
            else:
                missing = []
                if not raw_s2_path.exists():
                    missing.append(f"raw_s2: {raw_s2_path}")
                if not label_path.exists():
                    missing.append(f"label: {label_path}")
                print(f"Warning: Skipping pair - missing files: {', '.join(missing)}")

    def _find_simulated_file(self, expected_filename: str, s2_filename: str) -> Optional[Path]:
        """Find simulated file in the simulated directory structure.
        
        The simulated_sen1floods folder may have different structures:
        - Direct: simulated_sen1floods/simulated_X_Y_S2Hand.tif
        - Nested: simulated_sen1floods/simulated_train/simulated_X_Y_S2Hand.tif
        
        Args:
            expected_filename: Expected simulated filename (e.g., simulated_Ghana_123_S2Hand.tif)
            s2_filename: Original S2 filename for fallback
            
        Returns:
            Path to simulated file if found, None otherwise.
        """
        # Try direct path first
        direct_path = self.simulated_dir / expected_filename
        if direct_path.exists():
            return direct_path
        
        # Try in split subdirectory (e.g., simulated_train/)
        split_subdir = self.simulated_dir / f"simulated_{self.split_mapped}"
        split_path = split_subdir / expected_filename
        if split_path.exists():
            return split_path
        
        # Try alternative naming (in case different pattern was used)
        base_name = s2_filename.replace("S2Hand", "").strip("_")
        alt_filename = f"simulated_{base_name}.tif"
        alt_path = self.simulated_dir / alt_filename
        if alt_path.exists():
            return alt_path
            
        alt_split_path = split_subdir / alt_filename
        if alt_split_path.exists():
            return alt_split_path
        
        print(f"Warning: No simulated file found for {expected_filename}")
        return None

    def __len__(self) -> int:
        """Return number of valid pairs."""
        return len(self.pairs)

    def __iter__(self):
        """Iterate over pairs."""
        return iter(self.pairs)

    def get_pair(self, index: int) -> Dict[str, Path]:
        """Get a specific pair by index.
        
        Args:
            index: Index of the pair.
            
        Returns:
            Dictionary with 'raw_s2', 'simulated_s2', 'label' paths.
        """
        if index >= len(self.pairs):
            raise IndexError(f"Index {index} out of range for {len(self.pairs)} pairs")
        return self.pairs[index]

    def get_pairs(self, 
                  indices: Optional[List[int]] = None,
                  check_simulated: bool = True) -> List[Dict[str, Path]]:
        """Get multiple pairs.
        
        Args:
            indices: List of indices. If None, returns all pairs.
            check_simulated: If True, only returns pairs where simulated file exists.
            
        Returns:
            List of pair dictionaries.
        """
        if indices is None:
            indices = list(range(len(self.pairs)))
        
        result = []
        for idx in indices:
            pair = self.get_pair(idx)
            if check_simulated and pair["simulated_s2"] is None:
                continue
            result.append(pair)
        
        return result

    def load_pair(self, index: int) -> Dict[str, np.ndarray]:
        """Load all data for a pair: raw S2, simulated S2, and binary mask.
        
        Args:
            index: Index of the pair.
            
        Returns:
            Dictionary with:
            - 'raw_s2': ndarray (bands, height, width)
            - 'simulated_s2': ndarray (bands, height, width) or None
            - 'mask': ndarray (height, width) binary mask
        """
        pair = self.get_pair(index)
        
        # Load raw S2
        with rasterio.open(pair["raw_s2"]) as src:
            raw_s2 = src.read().astype(np.float32)
        
        # Load simulated S2 if available
        simulated_s2 = None
        if pair["simulated_s2"] is not None and pair["simulated_s2"].exists():
            try:
                with rasterio.open(pair["simulated_s2"]) as src:
                    simulated_s2 = src.read().astype(np.float32)
            except Exception as e:
                print(f"Warning: Could not load simulated file: {e}")
        
        # Load binary mask (ground truth)
        with rasterio.open(pair["label"]) as src:
            mask = src.read(1).astype(np.uint8)  # Binary mask
        
        return {
            "raw_s2": raw_s2,
            "simulated_s2": simulated_s2,
            "mask": mask,
        }

    def load_batch(self, indices: Optional[List[int]] = None) -> Dict[str, List]:
        """Load a batch of pairs.
        
        Args:
            indices: List of indices to load. If None, loads all pairs.
            
        Returns:
            Dictionary with lists of data:
            - 'raw_s2': list of ndarrays
            - 'simulated_s2': list of ndarrays
            - 'masks': list of ndarrays
            - 'pair_info': list of pair metadata
        """
        if indices is None:
            indices = list(range(len(self.pairs)))
        
        batch = {
            "raw_s2": [],
            "simulated_s2": [],
            "masks": [],
            "pair_info": [],
        }
        
        for idx in indices:
            try:
                data = self.load_pair(idx)
                pair_info = self.get_pair(idx)
                
                batch["raw_s2"].append(data["raw_s2"])
                batch["simulated_s2"].append(data["simulated_s2"])
                batch["masks"].append(data["mask"])
                batch["pair_info"].append({
                    "index": idx,
                    "filename": pair_info["s2_filename"],
                    "raw_path": str(pair_info["raw_s2"]),
                    "simulated_path": str(pair_info["simulated_s2"]) if pair_info["simulated_s2"] else None,
                    "label_path": str(pair_info["label"]),
                })
            except Exception as e:
                print(f"Error loading pair at index {idx}: {e}")
                continue
        
        return batch

    def get_location_from_filename(self, index: int) -> str:
        """Extract location name from S2 filename.
        
        Args:
            index: Index of the pair.
            
        Returns:
            Location name (e.g., 'Ghana', 'Mekong').
        """
        pair = self.get_pair(index)
        filename = pair["raw_s2"].name
        location = filename.split("_")[0]
        return location

    def get_split_info(self) -> Dict[str, int]:
        """Get split statistics.
        
        Returns:
            Dictionary with split name and pair counts.
        """
        pair_count = len(self.pairs)
        pairs_with_simulated = sum(
            1 for p in self.pairs 
            if p["simulated_s2"] is not None and p["simulated_s2"].exists()
        )
        
        return {
            "split": self.split,
            "total_pairs": pair_count,
            "pairs_with_simulated": pairs_with_simulated,
            "pairs_without_simulated": pair_count - pairs_with_simulated,
        }
