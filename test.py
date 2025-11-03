from dataset.types import Datum
from dataset.writer import write_dataset, write_parser_dict
from dataset.loader import load_tfr_dict, load_tfr_dataset
from dataset.utility import serialize_float_array
import os, sys, glob
from functools import partial
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This silences TensorFlow messages before it's imported
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # This disables all lower priority logging




def parser(feature_dict, shape_dict, example_proto):
    example = tf.io.parse_single_example(example_proto, feature_dict)
    img = tf.cast(example['image'], tf.float32)
    img = tf.reshape(img, shape_dict['image'])
    return img

def identity(dat):
    return dat


def image_decoder(dat):
    with open(dat.value, 'rb') as f:
        depth_map = np.load(f)
    return Datum(name=dat.name, value=depth_map)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    imgs = glob.glob(os.path.join(img_dir, "*"))

    gray_imgs = []
    parseables = []
    for img_pth in imgs:
        if img_pth.endswith('.json'):
            continue
        prs_list = []
        img_pth = os.path.join(img_pth, 'surface.npy')
        prs_list += [Datum(name='image', value=img_pth, decompress_fn=image_decoder, serialize_fn=serialize_float_array)]

        parseables.append(prs_list)

    write_parser_dict(parseables[-1], './', 'polymer.json')
    feature_dict, shape_dict = load_tfr_dict('./polymer.json')

    print(shape_dict)
    print(feature_dict)
    write_dataset(
        parseables,
        './',
        num_shards=10,
    )

    parser = partial(parser, feature_dict, shape_dict)
    ds = load_tfr_dataset(
        parser,
        './',
        '*.tfr*',
        verbose=2
    )

    for d in ds:
        print(d.shape)