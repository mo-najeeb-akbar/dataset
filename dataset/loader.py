from functools import partial
import tensorflow as tf
import json
import os

AUTOTUNE = tf.data.AUTOTUNE

def load_tfr_dataset(
        parser,
        feature_dict: dict[str, tf.io.FixedLenFeature],
        data_path: str,
        regex: str,
        shuffle: bool = True,
        cycle_length: int = 5,
        block_length: int = 5,
) -> tf.data.TFRecordDataset:
    """

    :param parser: function(feature_dict, example) to unpack the dataset
    :param feature_dict: decoding feature dictionary
    :param data_path: path to tfrecords
    :param regex: additional regex to look for files
    :param shuffle: should shuffle dataset on read
    :param cycle_length: controls the number of input records that are processed concurrently
    :param block_length: controls the number of record blocks to interleave
    :return:
    """
    parser = partial(parser, feature_dict)
    filenames_dataset = tf.data.Dataset.list_files(os.path.join(data_path, regex), shuffle=shuffle)
    dataset = filenames_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=AUTOTUNE
    )
    return dataset


def load_tfr_dict(
        json_path: str
) -> dict[str, tf.io.FixedLenFeature]:
    with open(json_path, 'r') as f:
        js_dict = json.load(f)
        tf_dict = {}

        for k, v in js_dict.items():
            if v['type'] == 'str':
                type_ = tf.string
                tf_dict[k] = tf.io.FixedLenFeature([], type_)
            else:
                type_ =  tf.float32
                num_vals_list = v['shape'].replace('(','').replace(')','').split(',')
                num_vals = 1
                [num_vals := num_vals * int(k) for k in num_vals_list]
                tf_dict[k] = tf.io.FixedLenFeature([num_vals], type_)
        return tf_dict