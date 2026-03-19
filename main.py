"""Entry point for TerraSat drift analysis experiments."""

import argparse

from terra_sat_drift.pipeline import DriftPipeline


def main() -> None:
    """Run model validation and bundled drift-analysis examples."""
    parser = argparse.ArgumentParser(
        description="TerraSat drift analysis using configurable TerraMind backbones."
    )
    parser.add_argument(
        "--backbone-size",
        choices=["tiny", "small", "base", "large"],
        default="large",
        help="TerraMind backbone size (default: large). Smaller models are faster but less accurate.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classification classes (default: 10).",
    )
    parser.add_argument(
        "--embedding-raw-dir",
        default="tiff_folder/raw_tiff_update",
        help="Source folder for embedding-based comparisons.",
    )
    parser.add_argument(
        "--embedding-simulated-dir",
        default="tiff_folder/simulated_custom_l2_tiff",
        help="Target folder for embedding-based comparisons.",
    )
    parser.add_argument(
        "--classification-raw-dir",
        default="tiff_folder/raw_tiff_update",
        help="Source folder for classification-based comparisons.",
    )
    parser.add_argument(
        "--classification-simulated-dir",
        default="tiff_folder/simulated_custom_l2_tiff",
        help="Target folder for classification-based comparisons.",
    )
    parser.add_argument(
        "--file1-pattern",
        default="BANDS_RES-GRID",
        help="Substring in source filenames used for counterpart matching.",
    )
    parser.add_argument(
        "--file2-pattern",
        default="PHISAT2-BANDS-GRID",
        help="Replacement substring in target filenames used for counterpart matching.",
    )
    parser.add_argument(
        "--suffix",
        default=".tiff",
        help="File suffix to scan when matching files.",
    )

    args = parser.parse_args()

    print(
        f"Starting DriftPipeline with backbone_size={args.backbone_size}, "
        f"num_classes={args.num_classes}"
        f"\nEmbedding comparison: {args.embedding_raw_dir} -> {args.embedding_simulated_dir}"
        f"\nClassification comparison: {args.classification_raw_dir} -> {args.classification_simulated_dir}"
    )
    pipeline = DriftPipeline(num_classes=args.num_classes, backbone_size=args.backbone_size)
    pipeline.validate()
    pipeline.run_examples(
        embedding_raw_dir=args.embedding_raw_dir,
        embedding_simulated_dir=args.embedding_simulated_dir,
        classification_raw_dir=args.classification_raw_dir,
        classification_simulated_dir=args.classification_simulated_dir,
        file1_pattern=args.file1_pattern,
        file2_pattern=args.file2_pattern,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()