#!/usr/bin/env python3
import argparse
import datetime
import functools
import os
import re
import sys

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# TODO:
#from omr_dataset import OMRDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 1710

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=tf.float32)

        hidden = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same")(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        hidden = tf.keras.layers.MaxPool2D((2, 2))(hidden)
        hidden = tf.keras.layers.Dropout(0.4)(hidden)

        hidden = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same")(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        hidden = tf.keras.layers.MaxPool2D((2, 2))(hidden)
        hidden = tf.keras.layers.Dropout(0.4)(hidden)

        hidden = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same")(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        hidden = tf.keras.layers.MaxPool2D((2, 2))(hidden)
        hidden = tf.keras.layers.Dropout(0.4)(hidden)

        hidden = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same")(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        hidden = tf.keras.layers.MaxPool2D((2, 2))(hidden)
        hidden = tf.keras.layers.Dropout(0.4)(hidden)

        hidden = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(hidden)
        new_shape = (IMAGE_WIDTH // 16, (IMAGE_HEIGHT // 16) * 128)
        hidden = tf.keras.layers.Reshape(new_shape)(hidden)

        hidden = tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(512, return_sequences=True)
        )(hidden)
        
        hidden = tf.keras.layers.Dropout(0.5)(hidden)

        logits = tf.keras.layers.Dense(1 + len(OMRDataset.MARKS), activation=None)(hidden)

        super().__init__(inputs=inputs, outputs=logits)

        self.compile(optimizer=tf.optimizers.Adam(clipnorm=0.001),
                     loss=self.ctc_loss,
                     metrics=[OMRDataset.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels, logits):
        
        logits = tf.RaggedTensor.from_tensor(logits)

        loss = tf.nn.ctc_loss(
            labels=tf.cast(gold_labels.to_sparse(), tf.int32),
            logits=logits.to_tensor(),
            label_length=None,
            logit_length=tf.cast(logits.row_lengths(), tf.int32),
            logits_time_major=False,
            unique=None,
            blank_index=-1,
            name="ctc_loss"
        )

        return tf.reduce_mean(loss)

    def ctc_decode(self, logits):

        logits = tf.RaggedTensor.from_tensor(logits)

        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(logits.to_tensor(), [1, 0, 2]),
            sequence_length=tf.cast(logits.row_lengths(), tf.int32),
            blank_index=-1
        )

        predictions = tf.RaggedTensor.from_sparse(decoded[0])

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


def main(args: argparse.Namespace) -> None:

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))


    omr = OMRDataset()

    # based on https://keras.io/examples/vision/handwriting_recognition/
    def resize_image(image):
        image = tf.image.resize_with_pad(image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
        
        return image

    def create_dataset(name):
        def prepare_example(example):
            new_input = resize_image(example["image"])
            mark = example["marks"]
            return new_input, mark

        dataset = getattr(omr, name).map(prepare_example)
        dataset = dataset.shuffle(200, seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_test():
        def prepare_example(example):
            new_input = resize_image(example["image"])
            return new_input

        dataset = getattr(omr, "test").map(prepare_example)
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_test()

    model = Model(args)
    model.summary()
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
