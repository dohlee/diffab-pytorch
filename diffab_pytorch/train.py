import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import warnings

from pytorch_lightning.callbacks import LearningRateMonitor
from diffab_pytorch import DiffAb
from diffab_pytorch.data import DiffAbDataModule

warnings.filterwarnings("ignore")


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta", required=True, help="Metadata file for train/validation data"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing PDB files"
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.1,
        help="Proportion of validation data to use.",
    )
    parser.add_argument("--cdrs", nargs="+", type=str, required=True)
    parser.add_argument("-b", "--bsz", type=int, default=128, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=60, help="Number of epochs to train"
    )
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Don't use wandb for logging",
    )
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = False

    args = parse_argument()
    pl.seed_everything(args.seed)

    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            project="deepab-pytorch",
            entity="dohlee",
            config=args,
        )

    # encoder parameters
    d_residue_emb = 128
    d_pair_emb = 64
    # IPA parameters
    n_ipa_layers = 6
    d_scalar_per_head = 32
    n_head = 8
    n_query_point_per_head = 8
    n_value_point_per_head = 8

    model = DiffAb(
        d_residue_emb,
        d_pair_emb,
        n_ipa_layers,
        d_scalar_per_head,
        n_query_point_per_head,
        n_value_point_per_head,
        n_head,
    )

    meta = pd.read_csv(args.meta).sample(frac=1, random_state=args.seed)
    train_df = meta.iloc[: int(len(meta) * (1 - args.val_pct))]
    val_df = meta.iloc[int(len(meta) * (1 - args.val_pct)) :]

    dm = DiffAbDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        data_dir=args.data_dir,
        cdrs_to_generate=args.cdrs,
        batch_size=args.bsz,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
