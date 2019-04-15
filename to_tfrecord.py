# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import glob
import sys
from PIL import Image

import cityscape
import Cityscape.labels as Labels

_NUM_SHARDS = 4
types = cityscape.types
CITYSCAPE_DIR = cityscape.CITYSCAPE_DIR
CITYSCAPE_IMG_DIR = cityscape.CITYSCAPE_IMG_DIR
CITYSCAPE_ANNO_DIR = cityscape.CITYSCAPE_ANNO_DIR

SAVED_IMG_FILES = cityscape.SAVED_IMG_FILES
SAVED_ANNO_FILES = cityscape.SAVED_ANNO_FILES


tfrecord_file = os.path.join(CITYSCAPE_DIR, 'tfrecord')

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_example(raw_img_data, anno_img_data, filename, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(raw_img_data),
        'image/anno': bytes_feature(anno_img_data),
        'image/filename': bytes_feature(filename),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

def to_tfrecord(index):
    # write the image and annotation to the tfrecord file
    if not os.path.exists(tfrecord_file):
        os.mkdir(tfrecord_file)

    img_file_list = SAVED_IMG_FILES[index]
    anno_file_list = SAVED_ANNO_FILES[index]

    f_img = open(img_file_list, 'r')
    f_anno = open(anno_file_list, 'r')
    img_filenames = f_img.readlines()
    anno_filenames = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in img_filenames]
    anno_filenames = [filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png') for filename in anno_filenames]

    # print(anno_filenames)

    assert len(img_filenames) == len(anno_filenames)

    num_per_shard = num_per_shard = int(math.ceil(len(anno_filenames) / _NUM_SHARDS))

    for shard_id in range(_NUM_SHARDS):
        output_tfrecord_filename = os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' %(types[index], shard_id, _NUM_SHARDS - 1))

        with tf.python_io.TFRecordWriter(output_tfrecord_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(anno_filenames))
            for i in range(start_ndx, end_ndx):
                try:
                    sys.stdout.write("\r>> Convert images %d/%d shard %d" % (i + 1, len(anno_filenames), shard_id))
                    sys.stdout.flush()

                    raw_img_data = Image.open(img_filenames[i].strip())
                    raw_img_data_np = np.array(raw_img_data)
                    height, width, _ = raw_img_data_np.shape

                    anno_data = Image.open(anno_filenames[i].strip())
                    anno_data_np = np.array(anno_data)
                    seg_height, seg_width = anno_data_np.shape
                    anno_data_np = Labels.id_to_trainId_map_func(anno_data_np)
                    anno_data_np = np.array(anno_data_np, dtype=np.uint8)
                    anno_data = Image.fromarray(anno_data_np)
                    assert seg_height == height
                    assert seg_width == width
                    raw_img_data = raw_img_data.tobytes()
                    anno_data = anno_data.tobytes()
                    # print(anno_data)
                    example = image_to_example(raw_img_data, anno_data, os.path.basename(img_filenames[i].strip()), height, width)
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print("Could not read: " + anno_filenames[i].strip())
                    print("Error: " + e)
                    print("Skip it\n")
    print("\n %s data is ok" % types[index])


if __name__ == '__main__':
    #[0, 1, 2]  =>  ['train', 'val', 'test']
    to_tfrecord(0)
    to_tfrecord(1)
    to_tfrecord(2)