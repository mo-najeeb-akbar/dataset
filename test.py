from dataset.types import ParseableDatum, SerializableDatum
from dataset.writer import write_dataset, write_parser_dict
from dataset.loader import load_tfr_dict, load_tfr_dataset
import os, sys, glob
import cv2
import numpy as np
import tensorflow as tf


def parser(feature_dict, example):
    example = tf.io.parse_single_example(example, feature_dict)
    img = tf.image.decode_image(example['image'])
    img = tf.image.grayscale_to_rgb(img)
    img_seg = tf.image.decode_image(example['seg'])
    return img, img_seg

def decode_function(parseable):
    res = []
    for k,v in parseable.references.items():
        if isinstance(v, str):
            img = cv2.imread(v, 0)
            res.append(SerializableDatum(name=k, value=img))
    for k, v in parseable.metadata.items():
        res.append(SerializableDatum(name=k, value=v))
    return res

if __name__ == "__main__":
    img_dir = sys.argv[1]
    label_imgs = glob.glob(os.path.join(img_dir, "pred*"))

    gray_imgs = []
    for imgpath in label_imgs:
        base_num = os.path.basename(imgpath).split('.')[0].split('_')[-1]
        gray_img = os.path.join(img_dir, f'img_{base_num}.png')
        gray_imgs.append(gray_img)
    parseables = []
    for img_pth, seg_pth in zip(gray_imgs, label_imgs):
        pd = ParseableDatum(
            {'image': img_pth, 'seg': seg_pth}, {'val0': 1.0, 'val2': np.zeros((3,4))}
        )
        parseables.append(pd)

    write_parser_dict(parseables[-1], './', 'roots.json')
    feature_dict = load_tfr_dict('./roots.json')

    write_dataset(
        decode_function,
        parseables,
        './',
        num_shards=3,
        num_workers=3,
        verbose=1
    )

    ds = load_tfr_dataset(
        parser,
        feature_dict,
        './',
        '*.tfr*',
        verbose=2
    )

    for d in ds:
        print(d[0].shape)