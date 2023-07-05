import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class DiffAb(pl.LightningModule):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
