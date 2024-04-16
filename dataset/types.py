from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Datum:
    """
    Represents a unit of data.

    Attributes:
        value: (np.ndarray, flot, int) that can be serialized into bytes.
        name: str that will be used to create a serialization dictionary.
    """
    value: any = field(default=None)
    name: str = field(default_factory=str)

    def __post_init__(self):
        if not isinstance(self.value, (np.ndarray, int, float, str)):
            raise TypeError(
                f"Value must be an instance of np.ndarray, int, or float, got {type(self.value).__name__}")

    def __str__(self):
        return f'Name: {self.name} -- Value: {self.value}'
