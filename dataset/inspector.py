import tensorflow as tf
from typing import Callable, Dict
from datetime import datetime
import os


def profile_tfr_dataset(dataset: tf.data.Dataset, logdir: str, num_samples: int):
    new_logdir = os.path.join(logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tf.profiler.experimental.start(new_logdir)

    for batch in dataset.take(num_samples):
        pass

    tf.profiler.experimental.stop()


def analyse_tfr_dataset(dataset: tf.data.Dataset, processors: list[Callable]) -> list:
    # Each processor will return a value
    num_processors = len(processors)
    results = [[] for _ in range(num_processors)]

    for item in dataset:
        for k in range(num_processors):
            results[k].append(processors[k](item))

    return results


def inspect_tfr_dataset(dataset: tf.data.Dataset, criterion: Callable, num_samples: int) -> list:
    # Find all elements in this dataset satisfying the criteria
    samples = []
    samples_remaining = num_samples
    for item in dataset:
        if criterion(item):
            samples.append(item.copy())
            samples_remaining -= 1
        if samples_remaining == 0:
            break
    if samples_remaining > 0:
        print(f'Only discovered {len(samples)} samples out of {num_samples}.')

    return samples


def check_tfr_dataset(dataset: tf.data.Dataset, criteria: list[Callable]) -> bool:
    for item in dataset:
        for criterion in criteria:
            if not criterion(item):
                return False

    return True
