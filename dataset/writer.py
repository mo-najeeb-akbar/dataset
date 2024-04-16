from .types import Datum
from .utility import split_list, serialize
import tensorflow as tf
import numpy as np
import os
import json
import multiprocessing
import threading
import queue


class TFRecordWriterPar:
    def __init__(self, file_path):
        self.file_path = file_path
        self.q = queue.Queue()
        self.writer_thread = threading.Thread(target=self.write_to_tfrecord)
        self.writer_thread.start()

    def write_to_tfrecord(self):
        with tf.io.TFRecordWriter(self.file_path) as writer:
            while True:
                data = self.q.get()
                if data is None:  # None is used as a signal to stop.
                    self.q.task_done()
                    break
                writer.write(data)
                self.q.task_done()

    def add_data(self, processed_data):
        self.q.put(processed_data)

    def close(self):
        self.q.put(None)
        self.writer_thread.join()


def write_dataset(
        decode_func,
        data_refs: list[Datum],
        output_path: str,
        extra_identifiers: list[str] | None = None,
        num_shards: int = 1
) -> None:
    """

    :param decode_func: f(list[Datum]) -> list[Datum] | None
    :param data_refs: [{filename, type_str}, ...]
    :param output_path: location of folder where to write data
    :param extra_identifiers: list of strings appended to file names for more information
    :param num_shards: number of separate chunks to write data as
    :return:

    """
    extra_suffix = '' if extra_identifiers is None else '_' + '_'.join(extra_identifiers)
    output_file_pre = os.path.join(output_path, f'record{extra_suffix}_')

    def worker(data: list[Datum], writer: TFRecordWriterPar):
        serializable_units = decode_func(data)
        if serializable_units is not None:
            serial_dict = serialize(serializable_units)
            example_proto = tf.train.Example(features=tf.train.Features(feature=serial_dict))
            writer.add_data(example_proto.SerializeToString())

    writers = [TFRecordWriterPar(output_file_pre + f'{num}.tfrecord') for num in range(num_shards)]

    # Setup multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # map_async for non-blocking calls
        block_size = (len(data_refs) // num_shards) + 1
        [pool.apply_async(worker, args=(data, writers[idx // block_size])) for idx, data in enumerate(data_refs)]
        # Close pool and wait for work to finish
        pool.close()
        pool.join()

    for w in writers:
        w.close()



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
