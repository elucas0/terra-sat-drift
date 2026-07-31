# Standard WorldCover classes to simplified indices (0-10)
# Mapping based on ESA WorldCover 10m v200
WC_CLASS_MAPPING = {
    '10': 0,  # Tree cover
    '20': 1,  # Shrubland
    '30': 2,  # Grassland
    '40': 3,  # Cropland
    '50': 4,  # Built-up
    '60': 5,  # Bare / sparse vegetation
    '70': 6,  # Snow and ice
    '80': 7,  # Permanent water bodies
    '90': 8,  # Herbaceous wetland
    '95': 9,  # Mangroves
    '100': 10 # Moss and lichen
}

# Human-readable names, index-aligned with the values of WC_CLASS_MAPPING.
# Passing these as `class_names` gives per-class metrics keys like
# "IoU_built_up" instead of "IoU_50" / the ambiguous "IoU_100".
WC_CLASS_NAMES = [
    "tree_cover",      # 0
    "shrubland",       # 1
    "grassland",       # 2
    "cropland",        # 3
    "built_up",        # 4
    "bare_sparse_veg", # 5
    "snow_ice",        # 6
    "water",           # 7
    "herbaceous_wetland",  # 8
    "mangroves",       # 9
    "moss_lichen",     # 10
]

# Same classes, index-aligned, for figures rather than metric keys.
WC_CLASS_DISPLAY_NAMES = [
    "Tree cover",              # 0
    "Shrubland",               # 1
    "Grassland",               # 2
    "Cropland",                # 3
    "Built-up",                # 4
    "Bare / sparse vegetation",# 5
    "Snow and ice",            # 6
    "Permanent water",         # 7
    "Herbaceous wetland",      # 8
    "Mangroves",               # 9
    "Moss and lichen",         # 10
]

# The official ESA WorldCover v200 legend colours, index-aligned with the values
# of WC_CLASS_MAPPING. Using the product's own legend rather than an arbitrary
# qualitative colormap means the class colours are the ones every WorldCover
# figure uses, and keeps every plot in this repo consistent with the others.
WC_CLASS_COLORS = [
    "#006400",  # 0  tree cover
    "#ffbb22",  # 1  shrubland
    "#ffff4c",  # 2  grassland
    "#f096ff",  # 3  cropland
    "#fa0000",  # 4  built-up
    "#b4b4b4",  # 5  bare / sparse vegetation
    "#f0f0f0",  # 6  snow and ice
    "#0064c8",  # 7  permanent water
    "#0096a0",  # 8  herbaceous wetland
    "#00cf75",  # 9  mangroves
    "#fae6a0",  # 10 moss and lichen
]

# Colour for the ignore_index (-1) pixels written by the class remapping.
NO_LABEL_COLOR = "#000000"

# Share of labelled pixels per class (%), index-aligned, measured over all
# 253,228 patches of the clean WorldCover manifest by
# scripts/dataset_report/analyze_manifest.py. Used to derive class weights for
# the imbalanced-segmentation losses. The tail matters: the last three classes
# together are ~1.2% of pixels but 27% of a macro-averaged mIoU.
WC_CLASS_PIXEL_FREQ = [
    19.995,  # 0  tree cover
    6.006,   # 1  shrubland
    20.777,  # 2  grassland
    13.422,  # 3  cropland
    14.400,  # 4  built-up
    16.196,  # 5  bare / sparse vegetation
    0.497,   # 6  snow and ice
    6.233,   # 7  permanent water
    1.423,   # 8  herbaceous wetland
    0.521,   # 9  mangroves
    0.331,   # 10 moss and lichen
]