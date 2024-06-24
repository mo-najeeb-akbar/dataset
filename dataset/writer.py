from .types import Datum
from .utility import split_list
import tensorflow as tf
import numpy as np
import os
import json
from typing import Tuple, Callable
import multiprocessing


def process_chunk(
    data: list[list[Datum]],
    closures: list[Tuple[Callable, Callable]],
    output_file_pre: str,
    id_: int
):
    writer_tf = tf.io.TFRecordWriter(f'{output_file_pre}{id_}.tfrecord')
    for idx, dat_ in enumerate(data):
        serializable_units = [closures[d_idx][0](dat) for d_idx, dat in enumerate(dat_)]
        if serializable_units is not None:
            serial_dict = {sdat.name: closures[s_idx][1](sdat) for s_idx, sdat in enumerate(serializable_units)}
            example_proto = tf.train.Example(features=tf.train.Features(feature=serial_dict))
            writer_tf.write(example_proto.SerializeToString())
    writer_tf.close()


def write_dataset(
        data_refs: list[list[Datum]],
        closures: list[Tuple[Callable, Callable]],
        output_path: str,
        extra_identifiers: list[str] | None = None,
        num_shards: int = 1,
) -> None:
    """
    :param data_refs: datums to convert
    :param closures: [(decompressor, serializer), ...] for the datums
    :param output_path: location of folder where to write data
    :param extra_identifiers: list of strings appended to file names for more information
    :param num_shards: number of separate chunks to write data as
    :return:
    """
    sharded_data_refs = split_list(data_refs, num_shards)
    extra_suffix = '' if extra_identifiers is None else '_' + '_'.join(extra_identifiers)
    output_file_pre = os.path.join(output_path, f'record{extra_suffix}_')

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use pool.starmap to pass both the function and items to the worker
        pool.starmap(process_chunk, [(references, closures, output_file_pre, shard_id)
                                               for shard_id, references in enumerate(sharded_data_refs)])


def write_parser_dict(
    data: list[Datum],
    closures: list[Tuple[Callable, Callable]],
    output_path: str,
    output_name: str
) -> None:
    """

    :param data:
    :param closures:
    :param output_path:
    :param output_name:
    :return:
    """
    res = {}
    serializable_units = [closures[d_idx][0](dat) for d_idx, dat in enumerate(data)]
    for datum in serializable_units:
        k = datum.name
        v = datum.value

        if isinstance(v, np.ndarray):
            shape = v.shape
        else:
            shape = '[1]'
        res[k] = {
            'type': type(v).__name__,
            'shape': f'{shape}'
        }
    with open(os.path.join(output_path, output_name), 'w') as f:
        json.dump(res, f, indent=4)
