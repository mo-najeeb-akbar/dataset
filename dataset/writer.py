from .types import Datum
from .utility import split_list
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
        threading_backend = 'loky',
        verbose: int = 0
) -> None:
    """
    :param data_refs: datums to convert
    :param output_path: location of folder where to write data
    :param extra_identifiers: list of strings appended to file names for more information
    :param num_shards: number of separate chunks to write data as
    :param num_workers: number of processes to spin up
    :param threading_backend: threading backend to use: loky, multiprocessing, threading
    :param verbose: 0: silent, 10: chunk completion
    :return:
    """
    sharded_data_refs = split_list(data_refs, num_shards)
    extra_suffix = '' if extra_identifiers is None else '_' + '_'.join(extra_identifiers)
    output_file_pre = os.path.join(output_path, f'record{extra_suffix}_')

    def process_chunk(data: list[list[Datum]], id_: int):
        writer_tf = tf.io.TFRecordWriter(f'{output_file_pre}{id_}.tfrecord')
        for idx, dat_ in enumerate(data):
            serializable_units = [dat.decompress_fn(dat) for dat in dat_]
            if serializable_units is not None:
                serial_dict = {sdat.name: sdat.serialize_fn(sdat) for sdat in serializable_units}
                example_proto = tf.train.Example(features=tf.train.Features(feature=serial_dict))
                writer_tf.write(example_proto.SerializeToString())
        writer_tf.close()

    Parallel(n_jobs=num_workers, backend=threading_backend, verbose=verbose, pre_dispatch='all')(
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
    serializable_units = [dat.decompress_fn(dat) for dat in data]
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
