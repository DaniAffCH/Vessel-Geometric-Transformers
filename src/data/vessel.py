from dataclasses import dataclass, field

from torch import Tensor
from torch_geometric.data import Data


@dataclass
class Vessel(Data):  # type: ignore[misc]
    """
    Data class for handling vessel data.

    This class inherits from Data and adds print utilities.
    """

    wss: Tensor = field(default_factory=Tensor)
    pos: Tensor = field(default_factory=Tensor)
    pressure: Tensor = field(default_factory=Tensor)
    face: Tensor = field(default_factory=Tensor)
    inlet_index: Tensor = field(default_factory=Tensor)

    def __repr__(self) -> str:
        return (
            f"Vessel(wss={self.wss.shape}, "
            f"pos={self.pos.shape}, "
            f"pressure={self.pressure.shape}, "
            f"face={self.face.shape}, "
            f"inlet_index={self.inlet_index.shape})"
        )
