from dataclasses import dataclass, field

from torch import Tensor
from torch_geometric.data import Data

from src.utils.definitions import Category, Feature


@dataclass
class Vessel(Data):  # type: ignore[misc]
    """
    This data class represents a vessel with the following attributes:
    - wss (Tensor): Wall shear stress tensor.
    - pos (Tensor): Position tensor.
    - pressure (Tensor): Pressure tensor.
    - face (Tensor): Face tensor.
    - inlet_index (Tensor): Inlet index tensor.

    This class inherits from Data and adds print utilities.
    """

    wss: Tensor = field(default_factory=Tensor)
    pos: Tensor = field(default_factory=Tensor)
    pressure: Tensor = field(default_factory=Tensor)
    face: Tensor = field(default_factory=Tensor)
    inlet_index: Tensor = field(default_factory=Tensor)
    label: Category = field(default=Category.Single)

    def __repr__(self) -> str:
        return (
            f"Vessel(wss={self.wss.shape}, "
            f"pos={self.pos.shape}, "
            f"pressure={self.pressure.shape}, "
            f"face={self.face.shape}, "
            f"inlet_index={self.inlet_index.shape})"
            f"label={self.label})"
        )


def get_feature(vessel: Vessel, feature: Feature) -> Tensor:
    if feature == Feature.WSS:
        return vessel.wss
    elif feature == Feature.POS:
        return vessel.pos
    elif feature == Feature.PRESSURE:
        return vessel.pressure
    elif feature == Feature.FACE:
        return vessel.face
    elif feature == Feature.INLET_INDEX:
        return vessel.inlet_index
    else:
        raise Exception("Wrong feature passed")
