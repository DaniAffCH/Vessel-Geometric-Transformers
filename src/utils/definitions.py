from enum import Enum


class Category(Enum):
    Bifurcating = 0
    Single = 1

    def __repr__(self) -> str:
        return f"Category.{self.name}"
