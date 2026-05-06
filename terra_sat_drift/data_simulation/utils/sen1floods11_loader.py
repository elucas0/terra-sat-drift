"""Data loader for extracting S2 files from Sen1Floods11 dataset for simulation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import rasterio
import json



class Sen1Floods11S2Loader:
    """Loader for extracting S2 files from Sen1Floods11 dataset.
    
    This loader reads split CSV files and extracts the corresponding S2 file paths
    for simulation without requiring the full multi-modal dataset loading.
    """

    def __init__(
        self,
        root_path: Path | str,
        split: str = "train",
        dataset_type: str = "HandLabeled",
    ):
        """Initialize the S2 loader for Sen1Floods11.

        Args:
            root_path: Root path to the sen1floods dataset (e.g., datasets/sen1floods11).
            split: Dataset split - one of ['train', 'valid', 'test'].
            dataset_type: Type of dataset - 'HandLabeled' or 'WeaklyLabeled'.
        """
        self.root_path = Path(root_path) / "v1.1"
        self.split = split
        self.dataset_type = dataset_type

        # Validate split
        split_mapping = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
        self.split_mapped = split_mapping.get(split, split)

        # Paths
        self.split_file = self.root_path / f"splits/flood_{dataset_type.lower()}" / f"flood_{self.split_mapped}_data.csv"
        self.data_root = self.root_path / f"data/flood_events/{dataset_type}"
        self.s2_dir = self.data_root / "S2Hand"
        self.s1_dir = self.data_root / "S1Hand"
        self.metadata_file = self.root_path / "Sen1Floods11_Metadata.geojson"

        self._validate_paths()
        self._load_split()

    def _validate_paths(self) -> None:
        """Validate that required directories exist."""
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.s2_dir.exists():
            raise FileNotFoundError(f"S2 directory not found: {self.s2_dir}")
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

    def _load_split(self) -> None:
        """Load the split CSV file and extract S2 file paths."""
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

        # Extract S2 file paths by converting S1Hand to S2Hand
        self.s2_files = []
        for s1_file, _ in file_list:
            # Convert S1Hand filename to S2Hand
            s2_filename = s1_file.replace("S1Hand", "S2Hand")
            s2_path = self.s2_dir / s2_filename
            if s2_path.exists():
                self.s2_files.append(s2_path)
            else:
                print(f"Warning: S2 file not found: {s2_path}")

        self.s1_files = [self.s1_dir / s1_file for s1_file, _ in file_list]

    def __len__(self) -> int:
        """Return number of S2 files."""
        return len(self.s2_files)

    def __iter__(self):
        """Iterate over S2 file paths."""
        return iter(self.s2_files)

    def get_s2_files(self) -> List[Path]:
        """Get all S2 file paths.

        Returns:
            List of Path objects pointing to S2 TIF files.
        """
        return self.s2_files.copy()

    def load_s2_file(self, index: int) -> np.ndarray:
        """Load a single S2 file as numpy array.

        Args:
            index: Index of the file to load.

        Returns:
            Numpy array with shape (bands, height, width).
        """
        if index >= len(self.s2_files):
            raise IndexError(f"Index {index} out of range for {len(self.s2_files)} files")

        s2_path = self.s2_files[index]
        with rasterio.open(s2_path) as src:
            return src.read().astype(np.float32)

    def load_s2_metadata(self, index: int) -> dict:
        """Load metadata for a single S2 file.

        Args:
            index: Index of the file.

        Returns:
            Dictionary with file metadata.
        """
        if index >= len(self.s2_files):
            raise IndexError(f"Index {index} out of range for {len(self.s2_files)} files")

        s2_path = self.s2_files[index]
        with rasterio.open(s2_path) as src:
            return {
                "path": str(s2_path),
                "shape": src.shape,
                "count": src.count,
                "dtype": src.dtypes[0],
                "profile": src.profile.copy(),
                "tags": src.tags(),
            }

    def get_location_from_filename(self, index: int) -> str:
        """Extract location name from S2 filename.

        Args:
            index: Index of the file.

        Returns:
            Location name (e.g., 'Bolivia', 'Ghana').
        """
        filename = self.s2_files[index].name
        location = filename.split("_")[0]
        return location

    def load_batch(
        self, indices: Optional[List[int]] = None, return_paths: bool = False
    ) -> dict:
        """Load a batch of S2 files.

        Args:
            indices: List of indices to load. If None, loads all files.
            return_paths: Whether to include file paths in output.

        Returns:
            Dictionary with 'data' (list of arrays) and optionally 'paths'.
        """
        if indices is None:
            indices = list(range(len(self.s2_files)))

        batch = {"data": [], "paths": [] if return_paths else None}

        for idx in indices:
            try:
                data = self.load_s2_file(idx)
                batch["data"].append(data)
                if return_paths:
                    batch["paths"].append((idx, str(self.s2_files[idx])))
            except Exception as e:
                print(f"Error loading file at index {idx}: {e}")
                continue

        return batch

    def load_geojson_metadata(self):
        """Load and return the Sen1Floods11 metadata GeoJSON.

        Returns:
            dict: Parsed GeoJSON metadata.
            
        Raises:
            FileNotFoundError: If metadata file not found.
            ImportError: If geopandas required but not available.
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        with open(self.metadata_file) as f:
            return json.load(f)