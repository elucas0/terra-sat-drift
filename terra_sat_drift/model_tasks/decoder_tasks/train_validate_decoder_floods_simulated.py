import torch
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import cv2
import albumentations as A

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    root_dir = Path("/shared/home/elucas/datasets/sen1floods11_simulated")
    
    # Target the domain-adapted weights saved during the MMD/Distillation training
    domain_adapted_weights = "/shared/home/elucas/scratch/terra-sat-drift/terramind_tiny_uda_mmd.pth"

    # Load band statistics from JSON file
    stats_file = f"{root_dir}/v1.1/spectral_statistics.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Extract means and stds for all bands
    means = list([stats[f"band_{i+1}"]["mean"] for i in range(8)])
    stds = list([stats[f"band_{i+1}"]["std"] for i in range(8)])
    TARGET_SIZE = (1088, 1088)  # Must be divisible by patch size (16)

    def preprocess_mask(mask, **kwargs):
        clean_mask = np.full(mask.shape, -1, dtype=np.int64)
        clean_mask[mask == 0] = 0
        clean_mask[mask == 1] = 1
        return clean_mask

    transform = [
        A.Resize(width=TARGET_SIZE[0], height=TARGET_SIZE[1], interpolation=cv2.INTER_NEAREST),
        A.Lambda(mask=preprocess_mask),
        A.pytorch.ToTensorV2(),
    ]

    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=8,
        data_root=root_dir,
        
        # We use the same roots for train/val/test and select samples via the given split files
        train_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        train_label_data_root=Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),
        val_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        val_label_data_root=Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),
        test_data_root=root_dir / "v1.1/data/flood_events/HandLabeled/S2Hand",
        test_label_data_root=Path("/shared/home/elucas/datasets/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),

        # Split files
        train_split=root_dir / "v1.1/splits/flood_handlabeled/flood_train_data.txt",
        val_split=root_dir / "v1.1/splits/flood_handlabeled/flood_valid_data.txt",
        test_split=root_dir / "v1.1/splits/flood_handlabeled/flood_test_data.txt",
        
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        means=means,
        stds=stds,
        dataset_bands=[0, 1, 2, 3, 4, 5, 6, 7],
        output_bands=[0, 1, 2, 3, 4, 5, 6, 7],
        num_workers=4,
        download=False,
        use_metadata=True,
        rgb_indices=[2, 1, 0],
        num_classes=2,
    )

    # Must perfectly match the architecture structure of the saved MMD weights
    COMPARED_BANDS = {
        "S2L1C": {
            "B02": 0, "B03": 1, "B04": 2, 
            "B05": 3, "B06": 4, "B07": 5, "B08": 6,
        }
    }

    # 2. Instantiate SemanticSegmentationTask
    task = SemanticSegmentationTask(
        model="Unet",
        model_args={
            "decoder_channels": [256, 128, 64, 32, 16],
        },
       #  backbone="terramind_v1_tiny",
        # Set pretrained to False here to prevent the 13-band registry crash
        backbone_pretrained=False, 
        backbone_args={
            "modalities": ["S2L1C"],
            "bands": COMPARED_BANDS,
        },
        loss="ce",
        # A slightly higher learning rate (1e-3) is standard when training a 
        # randomly initialized decoder while the encoder is frozen
        lr=1e-3, 
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        class_names=["background", "flood"],
        # CRITICAL: Freeze the domain-aligned encoder so the MMD adaptation is not destroyed
        freeze_backbone=True, 
    )

    # 3. Safely Inject the Domain-Adapted Weights into the Backbone
    print(f"Loading domain-adapted weights from {domain_adapted_weights}...")
    state_dict = torch.load(domain_adapted_weights, map_location="cpu")
    
    # TerraTorch's SemanticSegmentationTask utilizes segmentation-models-pytorch (SMP) or similar underneath,
    # which wraps the backbone inside an 'encoder' or 'backbone' attribute.
    backbone_module = None
    if hasattr(task.model, 'encoder'):
        backbone_module = task.model.encoder
    elif hasattr(task.model, 'backbone'):
        backbone_module = task.model.backbone
    
    if backbone_module is not None:
        try:
            backbone_module.load_state_dict(state_dict, strict=True)
            print("Domain-adapted weights successfully loaded into the frozen backbone!")
        except RuntimeError as e:
            print("Strict loading failed, attempting prefix matching. Error details:", e)
            backbone_module.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: Could not locate 'encoder' or 'backbone' in task.model.")

    # 4. Configure Callbacks and Logger
    experiment_name = "terramind_v1_tiny_simulated_decoder_only_uda"
    output_dir = "/shared/home/elucas/scratch/terra-sat-drift/outputs/terramind_sen1floods_simulated"
    
    logger = WandbLogger(project="terra-sat-drift", name=experiment_name, save_dir=output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/{experiment_name}/checkpoints",
        filename="best-val_mIoU",
        monitor="val/mIoU",
        mode="max",
        save_top_k=3,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/mIoU",
        patience=30,
        mode="max",
        verbose=True,
    )

    # 5. Initialize Trainer and Execute
    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    print(f"Starting Decoder-Only Fine-Tuning for {experiment_name}...")
    trainer.fit(task, datamodule=datamodule)