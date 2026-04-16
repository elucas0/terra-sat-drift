from pathlib import Path

from terratorch.datasets import Sen1Floods11NonGeo
import rasterio

class CustomSen1Floods11(Sen1Floods11NonGeo):
    def __getitem__(self, index: int):
        output = super().__getitem__(index)
        
        path = self.image_files[index]
        with rasterio.open(path) as src:
            output["bbox"] = src.bounds # (minx, miny, maxx, maxy)

        output["timestamp_coords"] = self._get_date(index)
        
        location = Path(path).name.split("_")[0]
        meta_row = self.metadata[self.metadata["location"] == location]
        if not meta_row.empty:
            output["timestamp"] = pd.to_datetime(meta_row["s2_date"].item())

        # Returns {'image', 'mask', 'bbox', and 'timestamp'}
        return output