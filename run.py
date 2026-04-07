"""Example usage of separated classification and segmentation services."""

from terra_sat_drift import (
    TerraMindClassifier,
    TerraMindSegmenter,
    DriftAnalyzer,
    DriftPipeline,
)
from pathlib import Path

def example_classification_only():
    """Perform classification-only drift analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Classification-Only Drift Analysis")
    print("=" * 80)
    
    # Initialize classifier
    classifier = TerraMindClassifier(num_classes=10, backbone_size="base")
    
    # Get class prediction for a single image
    s2_file = Path("path/to/s2_image.tif")
    prediction = classifier.get_class_prediction(s2_file)
    
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['predicted_probability']:.4f}")
    print(f"Top 3 classes: {prediction['top_3_classes']}")

def example_segmentation_only():
    """Perform binary semantic segmentation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Semantic Segmentation Only")
    print("=" * 80)
    
    # Initialize segmenter (binary segmentation: background vs foreground)
    segmenter = TerraMindSegmenter(num_classes=2, backbone_size="base")
    
    # Validate model
    segmenter.validate_setup()
    
    # Segment a single image
    s2_file = Path("path/to/s2_image.tif")
    segmentation_result = segmenter.segment_image(s2_file)
    
    # Get segmentation map and probabilities
    seg_map = segmentation_result["segmentation"]  # (height, width)
    prob_map = segmentation_result["probability_map"]  # (height, width, 2)
    
    print(f"Segmentation shape: {seg_map.shape}")
    print(f"Probability map shape: {prob_map.shape}")
    
    # Compute metrics against ground truth
    import numpy as np
    ground_truth = np.load("path/to/ground_truth_mask.npy")
    metrics = segmenter.compute_segmentation_metrics(seg_map, ground_truth)
    
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

def example_combined_drift_analysis():
    """Perform comprehensive drift analysis with both tasks."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Combined Classification + Segmentation Drift Analysis")
    print("=" * 80)
    
    # Initialize both services
    classifier = TerraMindClassifier(num_classes=10, backbone_size="base")
    segmenter = TerraMindSegmenter(num_classes=2, backbone_size="base")
    
    # Initialize drift analyzer with both services
    analyzer = DriftAnalyzer(classifier, segmenter=segmenter)
    
    # File pairs
    raw_file = Path("path/to/raw_s2.tif")
    simulated_file = Path("path/to/simulated_s2.tif")
    ground_truth = np.load("path/to/ground_truth.npy")
    
    # Analyze ALL drift types: spectral + embedding + classification + segmentation
    comprehensive_drift = analyzer.analyze_drift_with_segmentation(
        raw_file, simulated_file, ground_truth
    )
    
    print("\nSpectral Drift:")
    print(f"  Avg mean difference: {comprehensive_drift['spectral_drift']['summary']['avg_mean_difference']:.4f}")
    
    print("\nEmbedding Drift:")
    print(f"  Cosine similarity: {comprehensive_drift['embedding_drift']['cosine_similarity']:.4f}")
    
    print("\nClassification Drift:")
    print(f"  Class flip: {comprehensive_drift['class_drift']['class_flip']}")
    
    print("\nSegmentation Drift:")
    seg_drift = comprehensive_drift['segmentation_drift']['drift_metrics']
    print(f"  IoU drift: {seg_drift['iou_drift']:+.4f}")
    print(f"  Dice drift: {seg_drift['dice_drift']:+.4f}")

def example_full_pipeline():
    """Use the DriftPipeline for end-to-end analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Using DriftPipeline for End-to-End Analysis")
    print("=" * 80)
    
    # Initialize pipeline (creates both classifier and segmenter)
    pipeline = DriftPipeline(
        num_classes=10, 
        backbone_size="base",
        enable_segmentation=True  # Set to False to skip segmentation
    )

    # Run Sen1Floods drift analysis
    pipeline.run_sen1floods_drift_analysis(
        sen1floods_root="./datasets/sen1floods11",
        simulated_dir="./tiff_folder/simulated_sen1floods",
        split="train",
        num_samples=10,
    )
    
def example_custom_setup():
    """Custom setup with specific model configurations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custom Configuration")
    print("=" * 80)
    
    import torch
    
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Classification for 10 classes
    classifier = TerraMindClassifier(
        num_classes=10,
        backbone_size="large",
        device=device
    )
    
    # Multi-class segmentation (e.g., water, vegetation, urban, other)
    segmenter = TerraMindSegmenter(
        num_classes=4,  # water, vegetation, urban, other
        backbone_size="large",
        device=device
    )
    
    # Setup analyzer
    analyzer = DriftAnalyzer(classifier, segmenter=segmenter)
    
    print(f"Classification model: {classifier.num_classes} classes")
    print(f"Segmentation model: {segmenter.num_classes} classes")


if __name__ == "__main__":
    print("\nTerra-Sat-Drift: Separated Services Examples")
    print("=" * 80)
    
    # Uncomment any example to run:
    # example_classification_only()
    # example_segmentation_only()
    # example_combined_drift_analysis()
    example_full_pipeline()
    # example_custom_setup()
    
    print("\nSee comments above to run specific examples.")
