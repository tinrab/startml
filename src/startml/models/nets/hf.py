from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification


class HFSequenceClassifier(nn.Module):
    """Hugging Face sequence classification model that returns plain logits."""

    def __init__(self, model_name: str, num_labels: int, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        if freeze_backbone:
            base = getattr(self.model, self.model.base_model_prefix)
            for param in base.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
