#!/usr/bin/env python3
import argparse
import os
import uuid
import tensorflow as tf

COLAB = False
if COLAB:
    import sys
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.insert(0,'/content/drive/MyDrive/Colab Notebooks')
    os.chdir('/content/drive/MyDrive/Colab Notebooks/')

from omr_dataset import OMRDataset
from vocabulary_semantic import SYMBOL_LIST

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=48, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

# estimated from the examples
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 1710

maxpools = [2, 2, 2, 2]
conv_filters = [64, 64, 128, 128]
assert len(maxpools) == len(conv_filters)

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=tf.float32)

        # craete block with convolution, batchnorm, relu and maxpool
        for block_index in range(len(conv_filters)):
            hidden = tf.keras.layers.Conv2D(conv_filters[block_index], 3, activation=None, padding="same")(inputs if block_index == 0 else hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            hidden = tf.keras.layers.MaxPool2D((maxpools[block_index], maxpools[block_index]))(hidden)
        
        
        # Define the Inception blocks
        inception_block_1 = tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(hidden)
        inception_block_1 = tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu')(inception_block_1)
        inception_block_1 = tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu')(inception_block_1)

        inception_block_2 = tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(hidden)
        inception_block_2 = tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu')(inception_block_2)

        inception_block_3 = tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(hidden)

        inception_block_4 = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(hidden)
        inception_block_4 = tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu')(inception_block_4)
        
        # Concatenate the output of the Inception blocks
        hidden = tf.keras.layers.concatenate([inception_block_1, inception_block_2, inception_block_3, inception_block_4], axis=-1)
        
        

        hidden = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(hidden)
        new_shape = (106,6*352)
        hidden = tf.keras.layers.Reshape(new_shape)(hidden)
        
        hidden = tf.keras.layers.Dropout(0.3)(hidden)

        hidden = tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(512, return_sequences=True)
        )(hidden)
        
        hidden = tf.keras.layers.Dropout(0.3)(hidden)

        logits = tf.keras.layers.Dense(1 + len(SYMBOL_LIST), activation=None)(hidden)

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

    experiment_id = uuid.uuid4()
    print(f"Current experiment id: {experiment_id}")
    args.logdir = os.path.join("logs_", f"{experiment_id}")

    omr = OMRDataset()

    def create_dataset(name):
        def prepare_example(example):
            image = tf.image.resize_with_pad(example["image"], target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
            return image, example["label"]

        dataset = getattr(omr, name)
        dataset = dataset.shuffle(1000, seed=args.seed) if name == "train" else dataset
        dataset = dataset.map(prepare_example)
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    model = Model(args)
    model.summary()
    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    with open(os.path.join(args.logdir, "omr_result.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(SYMBOL_LIST[mark] for mark in sequence), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
