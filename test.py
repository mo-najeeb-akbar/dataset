from dataset.types import Datum
from dataset.writer import write_dataset, write_parser_dict
from dataset.loader import load_tfr_dict, load_tfr_dataset
import os, sys, glob
import cv2
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This silences TensorFlow messages before it's imported
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # This disables all lower priority logging




def parser(feature_dict, example):
    example = tf.io.parse_single_example(example, feature_dict)
    img = tf.image.decode_image(example['image'])
    img = tf.image.grayscale_to_rgb(img)
    img_seg = tf.image.decode_image(example['seg'])
    return img, img_seg

def identity(dat):
    return dat


def image_decoder(dat):
    return Datum(name=dat.name, value=cv2.imread(dat.value, 0))


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
        prs_list = []
        prs_list += [Datum(name='image', value=img_pth, function=image_decoder)]
        prs_list += [Datum(name='seg', value=seg_pth, function=image_decoder)]
        prs_list += [Datum(name='val_0', value=1.23, function=identity)]
        prs_list += [Datum(name='val_1', value=2.22, function=identity)]

        parseables.append(prs_list)

    write_parser_dict(parseables[-1], './', 'roots.json')
    feature_dict = load_tfr_dict('./roots.json')

    write_dataset(
        parseables,
        './',
        num_shards=5,
        num_workers=10
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