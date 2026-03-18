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

    args = parser.parse_args()

    print(
        f"Starting DriftPipeline with backbone_size={args.backbone_size}, "
        f"num_classes={args.num_classes}"
    )
    pipeline = DriftPipeline(num_classes=args.num_classes, backbone_size=args.backbone_size)
    pipeline.validate()
    pipeline.run_examples()


if __name__ == "__main__":
    main()