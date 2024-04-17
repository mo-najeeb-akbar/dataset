from .types import Datum
from .utility import split_list, serialize
import tensorflow as tf
import numpy as np
import os
import json
from joblib import Parallel, delayed


def write_dataset(
        data_refs: list[list[Datum]],
        output_path: str,
        extra_identifiers: list[str] | None = None,
        num_shards: int = 1,
        num_workers: int = 1,
        verbose: int = 0
) -> None:
    """
    :param data_refs: datums to convert
    :param output_path: location of folder where to write data
    :param extra_identifiers: list of strings appended to file names for more information
    :param num_shards: number of separate chunks to write data as
    :param num_workers: number of processes to spin up
    :param verbose: {0: silent, 1: chunk completion, 2: datum completion}
    :return:
    """
    sharded_data_refs = split_list(data_refs, num_shards)
    extra_suffix = '' if extra_identifiers is None else '_' + '_'.join(extra_identifiers)
    output_file_pre = os.path.join(output_path, f'record{extra_suffix}_')

    def process_chunk(data: list[list[Datum]], id_: int):
        writer_tf = tf.io.TFRecordWriter(f'{output_file_pre}{id_}.tfrecord')
        for idx, dat_ in enumerate(data):
            serializable_units = [dat.function(dat) for dat in dat_]
            if serializable_units is not None:
                serial_dict = serialize(serializable_units)
                example_proto = tf.train.Example(features=tf.train.Features(feature=serial_dict))
                writer_tf.write(example_proto.SerializeToString())
            if verbose == 2:
                print(f'Processed datum number: {idx} in chunk {id_}.')
        if verbose == 1:
            print(f'Finished writing chunk number: {id_}.')
        writer_tf.close()
    Parallel(n_jobs=num_workers, backend='loky', verbose=10)(
        delayed(process_chunk)(references, shard_id)
            for shard_id, references in enumerate(sharded_data_refs)
    )


def write_parser_dict(
        data: list[Datum],
        output_path: str,
        output_name: str
) -> None:
    """

    :param data:
    :param output_path:
    :param output_name:
    :return:
    """
    res = {}
    for datum in data:
        k = datum.name
        v = datum.value

        if isinstance(v, str):
            shape = 'None'
            res[k] = {
                'type': type(v).__name__,
                'shape': f'{shape}'
            }
        else:
            if isinstance(v, np.ndarray):
                shape = v.shape
            else:
                shape = '1'
            res[k] = {
                'type': type(v).__name__,
                'shape': f'{shape}'
            }
    with open(os.path.join(output_path, output_name), 'w') as f:
        json.dump(res, f, indent=4)
