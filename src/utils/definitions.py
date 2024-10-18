from enum import Enum


class Category(Enum):
    Bifurcating = 0
    Single = 1

    def __repr__(self) -> str:
        return f"{self.name}"


class Feature(Enum):
    WSS = 0
    POS = 1
    PRESSURE = 2
    FACE = 3
    INLET_INDEX = 4

    def __repr__(self) -> str:
        return f"Feature.{self.name}"
