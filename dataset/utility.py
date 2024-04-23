from .types import Datum
import tensorflow as tf
import cv2
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def split_list(list1, k):
    n = len(list1)  # Assuming both lists are of the same length
    part_size = n // k
    remainder = n % k

    parts1 = []
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


def serialize_image(data: Datum):
    success, encoded_image = cv2.imencode('.jpg', data.value)
    encoded_bytes = encoded_image.tobytes()
    return _bytes_feature(encoded_bytes)
