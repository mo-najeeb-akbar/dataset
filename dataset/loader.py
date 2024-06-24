from typing import Tuple
import tensorflow as tf
import json
import os
import glob

AUTOTUNE = tf.data.AUTOTUNE

def load_tfr_dataset(
        parser,
        data_path: str,
        regex: str,
        shuffle: bool = True,
        cycle_length: int = 3,
        block_length: int = 3,
        verbose=0
) -> tf.data.Dataset:
    """

    :param parser: function(feature_dict, example) to unpack the dataset
    :param data_path: path to tfrecords
    :param regex: additional regex to look for files
    :param shuffle: should shuffle dataset on read
    :param cycle_length: controls the number of input records that are processed concurrently
    :param block_length: controls the number of record blocks to interleave
    :param verbose: {0: nothing, 1: print(len(files)), 2: print(files)
    :return:
    """
    reg_paths = os.path.join(data_path, regex)
    files = glob.glob(reg_paths)
    if verbose == 1:
        print(f'Loader is using: {len(files)} files.')
    if verbose == 2:
        print(f'Loader is using: {files}')

    filenames_dataset = tf.data.Dataset.list_files(reg_paths, shuffle=shuffle)
    dataset = filenames_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=AUTOTUNE
    )
    return dataset


def load_tfr_dict(
        json_path: str
) -> Tuple[dict[str, tf.io.FixedLenFeature], dict[str, list[int]]]:
    with open(json_path, 'r') as f:
        js_dict = json.load(f)
        tf_dict = {}
        tf_shapes = {}
        for k, v in js_dict.items():
            if v['type'] == 'str':
                type_ = tf.string
                tf_dict[k] = tf.io.FixedLenFeature([], type_)
            else:
                num_vals_list = v['shape'].replace('(', '').replace(')', '').replace(' ', '').split(',')
                tf_shapes[k] = [int(val_) for val_ in num_vals_list]
                type_ =  tf.float32

                num_vals = 1
                _ = [num_vals := num_vals * val_ for val_ in tf_shapes[k]]
                tf_dict[k] = tf.io.FixedLenFeature([num_vals], type_)
        return tf_dict, tf_shapes