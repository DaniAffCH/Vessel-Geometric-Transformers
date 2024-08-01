import lightning as L
import torch
from torch import Tensor, nn, optim

from config.dataclasses import BaselineConfig


# TODO implement the BaselineTransformer
class BaselineTransformer(L.LightningModule):  # type: ignore[misc]
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.linear = nn.LazyLinear(out_features=1)
        self.config = config

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x * mask.unsqueeze(-1)  # Apply mask to zero out padded values
        x = self.linear(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = torch.zeros(1)
        loss.requires_grad = True
        return loss  # TODO

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        self.log("val_loss", torch.zeros(1))
        return torch.zeros(1)  # TODO

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=0.0001)  # TODO
