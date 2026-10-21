from functools import partial

import pytest
import torch
from torch import nn

from startml.models import ClassifierModule
from startml.models.nets import MLP, SimpleCNN


@pytest.mark.parametrize("batch_size", [2, 8])
def test_mlp_shapes(batch_size: int) -> None:
    net = MLP(input_size=784, hidden_sizes=(32, 16), output_size=10)
    assert net(torch.randn(batch_size, 1, 28, 28)).shape == (batch_size, 10)


def test_cnn_shapes() -> None:
    net = SimpleCNN(in_channels=3, channels=(8, 16), output_size=5)
    assert net(torch.randn(4, 3, 32, 32)).shape == (4, 5)


class DictNet(nn.Module):
    """Mimics a Hugging Face model taking keyword inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.linear(input_ids * attention_mask)


def make_module(net: nn.Module, num_classes: int) -> ClassifierModule:
    return ClassifierModule(
        net=net, optimizer=partial(torch.optim.SGD, lr=0.1), num_classes=num_classes
    )


def test_tuple_batch() -> None:
    module = make_module(MLP(input_size=4, hidden_sizes=(8,), output_size=3), num_classes=3)
    loss, preds, targets = module.model_step((torch.randn(5, 4), torch.randint(0, 3, (5,))))
    assert loss.ndim == 0
    assert preds.shape == targets.shape == (5,)


def test_dict_batch() -> None:
    module = make_module(DictNet(), num_classes=3)
    batch = {
        "input_ids": torch.randn(5, 4),
        "attention_mask": torch.ones(5, 4),
        "labels": torch.randint(0, 3, (5,)),
    }
    loss, preds, _ = module.model_step(batch)
    assert loss.ndim == 0
    assert preds.shape == (5,)
    probs = module.predict_step(batch, 0)
    assert torch.allclose(probs.sum(-1), torch.ones(5))


def test_hparams_are_plain_data() -> None:
    module = make_module(MLP(input_size=4, hidden_sizes=(8,), output_size=3), num_classes=3)
    assert set(module.hparams) == {
        "num_classes",
        "scheduler_interval",
        "label_smoothing",
        "compile",
    }
