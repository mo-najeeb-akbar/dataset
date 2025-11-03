# dataset

A lightweight, functional tool for building custom TFRecord datasets with minimal boilerplate.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from dataset import Datum, write_dataset, write_parser_dict
from dataset import load_tfr_dict, load_tfr_dataset
from dataset import serialize_float_array, serialize_float_or_int
import numpy as np
import tensorflow as tf

# 1. Define your data with Datum objects
data = [
    [
        Datum(name='image', value=np.random.rand(256, 256, 3), serialize_fn=serialize_float_array),
        Datum(name='label', value=1.23, serialize_fn=serialize_float_or_int)
    ],
    # ... more samples
]

# 2. Write to TFRecords (parallel, multi-shard)
write_dataset(data, './output', num_shards=10)

# 3. Save schema for loading
write_parser_dict(data[0], './', 'schema.json')

# 4. Load and use
feature_dict, shape_dict = load_tfr_dict('./schema.json')

def parser(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_dict)
    img = tf.reshape(tf.cast(example['image'], tf.float32), shape_dict['image'])
    return img, example['label']

ds = load_tfr_dataset(parser, './output', '*.tfrecord')
```

## Core Concepts

**Datum**: A named data element with serialization logic
- `value`: Your data (numpy array, int, float, str)
- `name`: Field name in the TFRecord
- `serialize_fn`: How to serialize (use built-in helpers)
- `decompress_fn`: Optional preprocessing before serialization

**Built-in serializers**:
- `serialize_float_array` - for numpy arrays
- `serialize_float_or_int` - for single numbers
- `serialize_image` - for tensor images
- `serialize_string` - for text

## Features

- **Parallel multi-shard writing** - Fast dataset creation using all CPU cores
- **Automatic schema generation** - Save/load field types and shapes
- **Custom preprocessing** - Apply transforms before serialization
- **Type-safe API** - Full type hints for IDE autocomplete and validation
- **Simple imports** - Everything available from `dataset` package

## TODO
- Reduce functional overhead with closures (add helper utilities)
