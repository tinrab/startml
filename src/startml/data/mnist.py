from collections.abc import Sequence
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """MNIST, with the official train+test sets pooled and re-split into train/val/test.

    Use this as the reference for writing new data modules: download in `prepare_data`
    (runs on one process), build datasets in `setup` (runs on every process).
    """

    def __init__(
        self,
        data_dir: str | Path = "data/",
        train_val_test_split: Sequence[int] = (55_000, 5_000, 10_000),
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        MNIST(self.hparams["data_dir"], train=True, download=True)
        MNIST(self.hparams["data_dir"], train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        # `batch_size` in the config is global; split it across devices.
        if self.trainer is not None:
            if self.hparams["batch_size"] % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"batch size ({self.hparams['batch_size']}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.hparams["batch_size"] // self.trainer.world_size

        if self.data_train is not None:
            return
        train = MNIST(self.hparams["data_dir"], train=True, transform=self.transforms)
        test = MNIST(self.hparams["data_dir"], train=False, transform=self.transforms)
        self.data_train, self.data_val, self.data_test = random_split(
            ConcatDataset([train, test]),
            lengths=self.hparams["train_val_test_split"],
            generator=torch.Generator().manual_seed(self.hparams["seed"]),
        )

    def _loader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        assert dataset is not None, "call setup() first"
        num_workers = self.hparams["num_workers"]
        return DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            num_workers=num_workers,
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["persistent_workers"] and num_workers > 0,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)
