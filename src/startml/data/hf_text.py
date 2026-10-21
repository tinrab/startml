from pathlib import Path
from typing import cast

from datasets import DatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase


class HFTextClassificationDataModule(LightningDataModule):
    """Text classification from the Hugging Face Hub (default: GLUE SST-2).

    Batches are dicts (`input_ids`, `attention_mask`, `labels`), which `ClassifierModule`
    forwards to the network as keyword arguments.
    """

    def __init__(
        self,
        tokenizer_name: str = "distilbert/distilbert-base-uncased",
        dataset_name: str = "nyu-mll/glue",
        dataset_config: str | None = "sst2",
        text_field: str = "sentence",
        label_field: str = "label",
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str | None = None,
        max_length: int = 128,
        num_classes: int = 2,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizer_name)
        )
        self.collator = DataCollatorWithPadding(self.tokenizer)
        self.dataset: DatasetDict | None = None

    @property
    def num_classes(self) -> int:
        return self.hparams["num_classes"]

    def _load(self) -> DatasetDict:
        return load_dataset(
            self.hparams["dataset_name"],
            self.hparams["dataset_config"],
            cache_dir=self.hparams["cache_dir"],
        )

    def prepare_data(self) -> None:
        self._load()

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is not None:
            return
        hp = self.hparams

        def tokenize(batch: dict) -> dict:
            return self.tokenizer(
                batch[hp["text_field"]], truncation=True, max_length=hp["max_length"]
            )

        splits = [hp["train_split"], hp["val_split"], hp["test_split"]]
        raw = self._load()
        dataset = DatasetDict({s: raw[s] for s in splits if s is not None})
        dataset = dataset.map(tokenize, batched=True)
        if hp["label_field"] != "labels":
            dataset = dataset.rename_column(hp["label_field"], "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataset = dataset

    def _loader(self, split: str | None, shuffle: bool) -> DataLoader:
        assert self.dataset is not None, "call setup() first"
        if split is None:
            raise ValueError("no split configured for this stage")
        return DataLoader(
            # HF datasets implement the map-style dataset protocol.
            cast(Dataset, self.dataset[split]),
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collator,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.hparams["train_split"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.hparams["val_split"], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        # Many Hub datasets ship unlabeled test splits; fall back to validation.
        return self._loader(self.hparams["test_split"] or self.hparams["val_split"], shuffle=False)
