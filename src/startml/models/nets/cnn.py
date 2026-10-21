from collections.abc import Sequence

from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """Small conv net: `[conv-bn-gelu] x2 + maxpool` per stage, then global pooling and a linear head."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64),
        output_size: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))
