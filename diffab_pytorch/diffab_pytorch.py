import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class SingleAminoAcidEmbeddingMLP(nn.Module):
    def __init__(self, d_type):
        self.embed_type = nn.Embedding(20, d_type)

        d_feat = d_type + 20
        pass

    def forward(self, x):
        pass


class PairwiseEmbeddingMLP(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class InvariantPointAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class AminoAcidDenoising(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class CaCoordinateDenoising(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class OrientationDenoising(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


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
