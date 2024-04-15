from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class ParseableDatum:
    """
    Represents a unit of unprocessed data to be consumed by a processing function.

    Attributes:
        references: Dict[str, str] is a reference to some piece of data combined with a string that should allow the
        user to easily decode and id that string for later use
        metadata: Dict[(np.ndarray, int, float), str] | None is actual data combined with a string that can be used to id that
        data for later use
    """
    references: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[any, str] | None = field(default=None)

    def __post_init__(self):
        if not isinstance(self.references, dict):
            raise TypeError(
                f"references must be a dict, got {type(self.references).__name__}"
            )

        for k, v in self.references.items():
            if not (isinstance(k, str) and isinstance(v, str)):
                raise TypeError(
                    f"References can only be a Dict[str,str], got {type(k).__name__}", {type(v).__name__})

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError(
                    f"metadata must be a dict, got {type(self.metadata).__name__}"
                )

            for k, v in self.metadata.items():
                if not (isinstance(k, str) and isinstance(v, (np.ndarray, int, float))):
                    raise TypeError(
                        f"References can only be a Dict[(np.ndarray, int, float),str], got {type(k).__name__}",
                        {type(v).__name__})

    def __str__(self):
        return f'References: {self.references} -- Metadata: {self.metadata}'


@dataclass(frozen=True)
class SerializableDatum:
    """
    Represents a unit of processed data to be consumed by a serializing function.

    Attributes:
        value: (np.ndarray, flot, int) that can be serialized into bytes.
        name: str that will be used to create a serialization dictionary.
    """
    value: any = field(default=None)
    name: str = field(default_factory=str)

    def __post_init__(self):
        if not isinstance(self.value, (np.ndarray, int, float)):
            raise TypeError(
                f"Value must be an instance of np.ndarray, int, or float, got {type(self.value).__name__}")

    def __str__(self):
        return f'Name: {self.name} -- Value: {self.value}'
