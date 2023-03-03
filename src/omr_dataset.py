import tensorflow as tf
import numpy as np
from vocabulary_semantic import SYMBOL_LIST
import argparse

class OMRDataset:
    # Parse a single example from the dataset
    def parse(example: tf.Tensor):
        # Parse the example from a serialized example string
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.VarLenFeature(tf.int64)}
            )
        
        # Decode the image from a PNG string
        example["image"] = tf.io.decode_png(example["image"], channels=1)
        # Convert the image data type to float32
        example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
        # Convert the label data type to int32
        example["label"] = tf.cast(tf.sparse.to_dense(example["label"]), tf.int32)

        return example

    def __init__(self) -> None:
        # Initialize the size of each dataset
        self._train_size: int = 73066
        self._dev_size: int = 7307
        self._test_size: int =  7305

        # Load the datasets from TFRecord files
        self.train: tf.data.Dataset = tf.data.TFRecordDataset('train.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._train_size))
        self.dev: tf.data.Dataset = tf.data.TFRecordDataset('dev.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._dev_size))
        self.test: tf.data.Dataset = tf.data.TFRecordDataset('test.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._test_size))

    # Class to calculate the mean edit distance between two datasets
    class EditDistanceMetric(tf.metrics.Mean):
            def __init__(self, name: str = "edit_distance", dtype = None) -> None:
                super().__init__(name, dtype)

            # Update the metric state with a new set of true and predicted labels
            def update_state(self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor, sample_weight = None) -> None:
                # Calculate the edit distance between the true and predicted labels
                edit_distances = tf.edit_distance(y_pred.to_sparse(), y_true.to_sparse(), normalize=True)
                # Update the metric state with the new edit distances
                return super().update_state(edit_distances, sample_weight)

    # Static method to evaluate the prediction performance on a gold dataset
    @staticmethod
    def evaluate(gold_dataset, predictions):
        # Convert the gold dataset labels to strings
        gold_data = [[SYMBOL_LIST[symbol] for symbol in np.array(example["label"])] for example in gold_dataset]

        if len(predictions) != len(gold_data):
            raise RuntimeError(f"The predictions should have the same size as golden data: {len(predictions)} vs {len(gold_data)}")

        # NOTE: tensorflow uses gold data for normalising so normalised edit distance can be actually > 1
        edit_distance_metric = OMRDataset.EditDistanceMetric()
        gold_batches = [gold_data[i:i + 16] for i in range(0, len(gold_data), 16)]
        predictions_batches = [predictions[i:i + 16] for i in range(0, len(predictions), 16)]

        for gold_batch, prediction_batch in zip(gold_batches, predictions_batches):
            edit_distance_metric(
                tf.ragged.constant(gold_batch, tf.string),
                tf.ragged.constant(prediction_batch, tf.string),
            )

        return (100 * edit_distance_metric.result()).numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", default='omr_result.txt', type=str, help="Name of file with generated predictions.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    omr = OMRDataset()

    with open(args.predictions_file, 'r') as predictions_file:
        predictions = []
        for line in predictions_file:
            predictions.append(line.rstrip("\n").split())
        print(f"Predictions edit distance: {OMRDataset.evaluate(omr.test, predictions):.4f} %")


    