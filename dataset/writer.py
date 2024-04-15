from utility import ParseableDatum, split_two_lists, serialize
from joblib import Parallel, delayed
import tensorflow as tf
import os


def write_dataset(
        decode_func,
        data_refs: list[ParseableDatum],
        data_stores: list[ParseableDatum],
        output_path: str,
        extra_identifiers: list[str] | None = None,
        num_shards: int = 1,
        num_workers: int = 1,
        verbose: int = 0
) -> None:
    """

    :param decode_func: f(data_ref, data_store_elt) -> list[SerializableDatum] | None
    :param data_refs: [{filename, type_str}, ...]
    :param data_stores: [{data, type_str}, ...]
    :param output_path: location of folder where to write data
    :param extra_identifiers: list of strings appended to file names for more information
    :param num_shards: number of separate chunks to write data as
    :param num_workers: number of processes to spin up
    :param verbose: {0: silent, 1: chunk completion, 2: datum completion}
    :return:

    """
    sharded_data_refs, sharded_data_stores = split_two_lists(data_refs, data_stores, num_shards)
    extra_suffix = '' if extra_identifiers is not None else '_' + '_'.join(extra_identifiers)
    output_file_pre = os.path.join(output_path, f'record{extra_suffix}_')

    def process_chunk(refs: list[ParseableDatum], stores: list[ParseableDatum], id: int):
        writer_tf = tf.io.TFRecordWriter(f'{output_file_pre}{id}.tfrecord')
        for idx, (ref, store) in enumerate(zip(refs, stores)):
            serializable_units = decode_func(ref, store)
            if serializable_units is not None:
                serial_dict = serialize(serializable_units)
                example_proto = tf.train.Example(features=tf.train.Features(feature=serial_dict))
                writer_tf.write(example_proto.SerializeToString())
            if verbose == 2:
                print(f'Processed datum number: {idx} in chunk {id}.')
        if verbose == 1:
            print(f'Finished writing chunk number: {id}.')

        Parallel(n_jobs=num_workers)(
            delayed(process_chunk)(references, stores, shard_id)
                for shard_id, (references, stores)in enumerate(zip(sharded_data_refs, sharded_data_stores))
        )


