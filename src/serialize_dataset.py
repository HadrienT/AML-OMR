import tensorflow as tf 
import os
import fnmatch
import numpy as np

from vocabulary_semantic import SYMBOL_INT_VOCAB

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# NOTE: you can download the dataset here: https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz
root = './OMRdata'
count = 0

# Count the number of subdirectories (i.e., number of examples)
for path, subdirs, files in os.walk(root):
    if len(subdirs):
        print(path)
        count += len(subdirs)
    else: continue

print(f"Number of files: {count}")

# Split the examples into training, development, and test sets
train_count = int(count / 1.2)
dev_count = int((count - train_count) / 2)
test_count = count - train_count - dev_count

print(f"Train examples: {train_count}, dev examples: {dev_count}, test_examples: {test_count}")

# Initialize the TFRecord writer for the training set
writer = tf.io.TFRecordWriter('train.tfrecord')
phase = 'train'
written = 0

# Iterate through the subdirectories (examples)
for path, subdirs, files in os.walk(root):
    if len(subdirs) == 0:
        img_path = None
        label_path = None

        # Find the image and label files
        for name in files:
            # skip hidden files
            if name[0] == '.': continue
            if fnmatch.fnmatch(name, '*.png'):
                img_path = os.path.join(path, name)
            elif fnmatch.fnmatch(name, '*.semantic'):
                label_path = os.path.join(path, name)

        # If both image and label files were found, process the example
        if img_path is not None and label_path is not None:
            # Read the image and convert it to PNG format
            img_bytes = tf.io.gfile.GFile(img_path, 'rb').read()
            img = tf.io.decode_png(img_bytes, channels=1)
            # img = tf.cast(255 - img, tf.uint8) # NOTE: if we want it inverted -> 0 .. white, 1 .. black
            img = tf.io.encode_png(tf.cast(img, tf.uint8)).numpy()
            # Read the labels from the file and split them into a list
            with open(label_path, 'r') as label_file:
                labels = label_file.read().splitlines()[0].strip().split('\t')

                # Convert the labels to a list of integers
                label = np.array([*map(SYMBOL_INT_VOCAB.get, labels)])

                # Create a TensorFlow feature for the image and label
                feature = {
                    'image': _bytes_feature(img),
                    'label': _int64_feature(label)
                    }

                # Create a TensorFlow example using the features and write the example to the appropriate TFRecord file
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                written += 1

                # If the appropriate number of examples have been written to the current TFRecord file,
                # switch to the next phase (i.e., development or test set)
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