from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from glob import glob
import json

from dataset import HLS_US
from model import MACUNet

classes = [
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed/Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow/Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other"
]

with open('./data_splits/training_data.txt') as file:
    train_list = [line.rstrip() for line in file]
with open('./data_splits/validation_data.txt') as file:
    val_list = [line.rstrip() for line in file]
# Testing
# train_list = ['chip_002_060']
# val_list = ['chip_010_063']

def train():
    wandb_logger = WandbLogger(project="MNIST")

    model = MACUNet(6, len(classes))

    train_ds = HLS_US('./data/', train_list, num_classes=len(classes))
    train_loader = DataLoader(train_ds, batch_size=16)
    val_ds = HLS_US('./data/', val_list, num_classes=len(classes))
    val_loader = DataLoader(val_ds, batch_size=16)

    trainer = pl.Trainer(max_epochs=22, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader, ckpt_path=None)

if __name__ == '__main__':
    train()