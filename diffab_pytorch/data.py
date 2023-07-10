import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from diffab_pytorch.diffusion import (
    SequenceDiffuser,
    CoordinateDiffuser,
)


class DiffAbDataset(Dataset):
    def __init__(self, meta_df, T=100, s=0.01, beta_max=0.999):
        super().__init__()

        self.meta_df = meta_df
        self.records = meta_df.to_records()

        self.T = T

        self.seq_diffuser = SequenceDiffuser(
            T=self.T,
            s=s,
            beta_max=beta_max,
        )

        self.coord_diffuser = CoordinateDiffuser(
            T=self.T,
            s=s,
            beta_max=beta_max,
        )

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, i):
        record = self.records[i]
        ret = {}

        # sample timepoint t from 0..T-1
        t = torch.randint(0, self.T, (1,))
        ret["t"] = t

        #
        # sequence
        #
        seq = record.seq
        ret["seq"] = seq

        # add appropriate noise to sequence
        seq_t = self.seq_diffuser.diffuse_from_t0(record.seq, t=t)
        ret["seq_noised"] = seq_t

        seq_posterior_target = self.posterior_single_step(seq_t, record.seq, t=t)
        ret["seq_posterior_target"] = seq_posterior_target

        #
        # Ca coordinate
        #
        xyz = record.xyz
        ret["xyz"] = xyz

        # add appropriate noise to coordinate and keep track of that noise
        xyz_t, xyz_eps = self.coord_diffuser.diffuse_from_t0(record.xyz, t=t, return_eps=True)
        ret["xyz_noised"] = xyz_t
        ret["xyz_eps"] = xyz_eps

        return ret


class DiffAbDataModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
