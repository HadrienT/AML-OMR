import tensorflow as tf 
import os
import fnmatch
import numpy as np

from vocabulary_semantic import SYMBOL_INT_VOCAB

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

root = './OMRdata'
count = 0

for path, subdirs, files in os.walk(root):
    if len(subdirs):
        print(path)
        count += len(subdirs)
    else: continue

print(f"Number of files: {count}")

train_count = int(count / 1.2)
dev_count = int((count - train_count) / 2)
test_count = count - train_count - dev_count

print(f"Train examples: {train_count}, dev examples: {dev_count}, test_examples: {test_count}")

writer = tf.io.TFRecordWriter('train.tfrecord')
phase = 'train'
written = 0
for path, subdirs, files in os.walk(root):
    if len(subdirs) == 0:
        img_path = None
        label_path = None

        for name in files:
            # skip hidden files
            if name[0] == '.': continue
            if fnmatch.fnmatch(name, '*.png'):
                img_path = os.path.join(path, name)
            elif fnmatch.fnmatch(name, '*.semantic'):
                label_path = os.path.join(path, name)

        if img_path is not None and label_path is not None:
            img_bytes = tf.io.gfile.GFile(img_path, 'rb').read()
            img = tf.io.decode_png(img_bytes, channels=1)
            # img = tf.cast(255 - img, tf.uint8) # NOTE: if we want it inverted -> 0 .. white, 1 .. black
            img = tf.io.encode_png(tf.cast(img, tf.uint8)).numpy()
            with open(label_path, 'r') as label_file:
                labels = label_file.read().splitlines()[0].strip().split('\t')
            label = np.array([*map(SYMBOL_INT_VOCAB.get, labels)])
            feature = {
                'image': _bytes_feature(img),
                'label': _int64_feature(label)
                }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            written += 1

            if written == train_count:
                writer.close()
                writer = tf.io.TFRecordWriter('dev.tfrecord')
                written = 0
                phase = 'dev'
            elif written == dev_count and phase == 'dev':
                writer.close()
                writer = tf.io.TFRecordWriter('test.tfrecord')
                written = 0
                phase = 'test'
            elif written == test_count and phase == 'test':
                writer.close()
                break
