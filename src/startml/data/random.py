from collections.abc import Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class RandomDataModule(LightningDataModule):
    """Synthetic classification data: each class is a random prototype plus Gaussian noise.

    Learnable but needs no download, so it's used for tests and smoke runs.
    """

    def __init__(
        self,
        input_shape: Sequence[int] = (1, 28, 28),
        num_classes: int = 10,
        train_size: int = 512,
        val_size: int = 128,
        test_size: int = 128,
        noise: float = 1.0,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: TensorDataset | None = None
        self.data_val: TensorDataset | None = None
        self.data_test: TensorDataset | None = None

    @property
    def num_classes(self) -> int:
        return self.hparams["num_classes"]

    def _make(self, size: int, generator: torch.Generator, centers: torch.Tensor) -> TensorDataset:
        y = torch.randint(0, self.num_classes, (size,), generator=generator)
        noise = torch.randn(size, *centers.shape[1:], generator=generator)
        return TensorDataset(centers[y] + self.hparams["noise"] * noise, y)

    def setup(self, stage: str | None = None) -> None:
        if self.data_train is not None:
            return
        generator = torch.Generator().manual_seed(self.hparams["seed"])
        centers = torch.randn(self.num_classes, *self.hparams["input_shape"], generator=generator)
        self.data_train = self._make(self.hparams["train_size"], generator, centers)
        self.data_val = self._make(self.hparams["val_size"], generator, centers)
        self.data_test = self._make(self.hparams["test_size"], generator, centers)

    def _loader(self, dataset: TensorDataset | None, shuffle: bool) -> DataLoader:
        assert dataset is not None, "call setup() first"
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
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
