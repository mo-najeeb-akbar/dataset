"""
dataset: A lightweight TFRecord dataset writing and loading library.

This package provides a functional approach to building custom TFRecord datasets
with minimal boilerplate. It includes tools for writing, loading, and serializing
data with parallel multi-shard support.
"""

from .types import Datum, DatumValue, SerializeFn, DecompressFn
from .writer import write_dataset, write_parser_dict
from .loader import load_tfr_dataset, load_tfr_dict
from .utility import (
    serialize_float_array,
    serialize_float_or_int,
    serialize_image,
    serialize_string,
)

__all__ = [
    # Core types
    'Datum',
    'DatumValue',
    'SerializeFn',
    'DecompressFn',
    
    # Writer functions
    'write_dataset',
    'write_parser_dict',
    
    # Loader functions
    'load_tfr_dataset',
    'load_tfr_dict',
    
    # Serialization utilities
    'serialize_float_array',
    'serialize_float_or_int',
    'serialize_image',
    'serialize_string',
]

__version__ = '0.1.0'
