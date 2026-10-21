from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Fully connected network; flattens its input first."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: Sequence[int] = (256, 256),
        output_size: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = input_size
        for hidden in hidden_sizes:
            layers += [
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_features = hidden
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
