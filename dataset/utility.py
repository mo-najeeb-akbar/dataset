from .types import Datum
import tensorflow as tf
from typing import TypeVar, List, Sequence

T = TypeVar('T')


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: Sequence[float]) -> tf.train.Feature:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def split_list(list1: List[T], k: int) -> List[List[T]]:
    """Split a list into k roughly equal parts."""
    n = len(list1)
    part_size = n // k
    remainder = n % k

    parts1: List[List[T]] = []
    taken = 0
    for i in range(k):
        next_taken = taken + part_size + (1 if i < remainder else 0)
        parts1.append(list1[taken:next_taken])
        taken = next_taken

    return parts1


def serialize_float_array(data: Datum) -> tf.train.Feature:
    return _float_feature(data.value.flatten())


def serialize_float_or_int(data: Datum) -> tf.train.Feature:
    return _float_feature([data.value])


def serialize_image(data: Datum) -> tf.train.Feature:
    return _bytes_feature(tf.io.serialize_tensor(data.value).numpy())


def serialize_string(data: Datum) -> tf.train.Feature:
    encoded_bytes = data.value.encode('utf-8')
    return _bytes_feature(encoded_bytes)
