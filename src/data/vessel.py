from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class Vessel(Data):  # type: ignore[misc]
    """
    Data class for handling vessel data.

    This class inherits from Data and is used to
    handle vessel data.
    """

    wss: Tensor = field(default_factory=Tensor)
    pos: Tensor = field(default_factory=Tensor)
    pressure: Tensor = field(default_factory=Tensor)
    face: Tensor = field(default_factory=Tensor)
    inlet_index: Tensor = field(default_factory=Tensor)

    def __post_init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"Vessel(wss={self.wss.shape}, "
            f"pos={self.pos.shape}, "
            f"pressure={self.pressure.shape}, "
            f"face={self.face.shape}, "
            f"inlet_index={self.inlet_index.shape})"
        )

    def to_tensor(self) -> Tensor:
        """
        Convert the Vessel object into a single tensor.

        Returns:
            Tensor: A single tensor containing all the attributes.
        """
        return torch.cat(
            [self.wss, self.pos, self.pressure, self.face, self.inlet_index],
            dim=0,
        )
