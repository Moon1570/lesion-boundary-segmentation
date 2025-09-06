# Training config
from config.segmentor import Segmentor
import pytorch_lightning as pl
import os
import torch
from torch.utils.data import DataLoader
from module.model.proposed_net import DermoMamba
from datasets.datasets import ISICLoader

# Initialize model
model = DermoMamba()
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Path to your preprocessed data (adjust path relative to DermoMamba-main directory)
data_dir = "../data/ISIC2018_proc"
splits_dir = "../splits"

# Create datasets
train_dataset = ISICLoader(
    data_dir=data_dir,
    splits_dir=splits_dir,
    split="train",
    image_size=384,
    is_train=True
)

val_dataset = ISICLoader(
    data_dir=data_dir,
    splits_dir=splits_dir,
    split="val",
    image_size=384,
    is_train=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# Create checkpoint directory
os.makedirs('./weight/ISIC2018/', exist_ok=True)

# Setup callbacks
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint(
    './weight/ISIC2018/', 
    filename="dermomamba_ckpt_{val_dice:.4f}",
    monitor="val_dice", 
    mode="max", 
    save_top_k=1,
    verbose=True, 
    save_weights_only=True,
    auto_insert_metric_name=False
)

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_dice",
    mode="max",
    patience=15,
    verbose=True
)

progress_bar = pl.callbacks.TQDMProgressBar()

PARAMS = {
    "benchmark": True, 
    "enable_progress_bar": True,
    "logger": True,
    "callbacks": [check_point, early_stopping, progress_bar],
    "log_every_n_steps": 1, 
    "num_sanity_val_steps": 0, 
    "max_epochs": 200,
    "precision": 16,
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "devices": 1 if torch.cuda.is_available() else None,
}

trainer = pl.Trainer(**PARAMS)
segmentor = Segmentor(model=model)

# Print training info
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"Training DermoMamba on ISIC2018 dataset")
print(f"Using Guide Fusion Loss + Cross-Scale Mamba Blocks + CBAM attention")
print("-" * 60)

# CHECKPOINT_PATH = ""
# segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)

# Training
trainer.fit(segmentor, train_loader, val_loader)