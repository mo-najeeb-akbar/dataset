from dataclasses import dataclass, field
import numpy as np
from typing import Callable, TypeVar, Optional

T = TypeVar('T')


def identity(x: T) -> T:
    return x


def get_function_name(func: Optional[Callable]) -> str:
    # Attempt to retrieve the function's name, handling cases where it may not be directly available
    if func is None:
        return ''
    return getattr(func, '__name__', repr(func))


@dataclass(frozen=True)
class Datum:

    value: any = field(default=None)
    name: str = field(default_factory=str)
    # decompress_fn: Callable[[T], T] = field(default=identity)
    # serialize_fn: Callable[[T], bytes] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.value, (np.ndarray, int, float, str, tuple)):
            raise TypeError(
                f"Value must be an instance of np.ndarray, int, float, str, or tuple -- got {type(self.value).__name__}")

    def __str__(self):
        return (f'Name: {self.name}\n'
                f'    Value: {self.value}\n')
                # f'    Decompress Fn: {get_function_name(self.decompress_fn)}\n'
                # f'    Serialze Fn: {get_function_name(self.serialize_fn)}')
