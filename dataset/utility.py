from .types import SerializableDatum
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


def split_two_lists(list1, list2, k):
    n = len(list1)  # Assuming both lists are of the same length
    part_size = n // k
    remainder = n % k

    parts1 = []
    parts2 = []
    taken = 0
    for i in range(k):
        next_taken = taken + part_size + (1 if i < remainder else 0)
        parts1.append(list1[taken:next_taken])
        parts2.append(list2[taken:next_taken])
        taken = next_taken

    return parts1, parts2


def serialize(data: list[SerializableDatum]) -> dict[str, tf.train.Feature]:
    result_dict = {}
    for datum in data:
        if isinstance(datum.value, np.ndarray):
            if datum.value.dtype == np.uint8:
                success, encoded_image = cv2.imencode('.jpg', datum.value)
                encoded_bytes = encoded_image.tobytes()
                result_dict[datum.name] = _bytes_feature(encoded_bytes)
            else:
                result_dict[datum.name] = _float_feature(datum.value.flatten())
        else:
            result_dict[datum.name] = _float_feature([datum.value])
    return result_dict