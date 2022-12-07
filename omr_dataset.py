import tensorflow as tf

class OMRDataset:
    def parse(example: tf.Tensor):
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.VarLenFeature(tf.int64)}
            )
        
        example["image"] = tf.io.decode_png(example["image"], channels=1)
        example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
        example["label"] = tf.cast(tf.sparse.to_dense(example["label"]), tf.int32)

        return example

    def __init__(self) -> None:
        self._train_size: int = 73066
        self._dev_size: int = 7307
        self._test_size: int =  7305

        self.train: tf.data.Dataset = tf.data.TFRecordDataset('train.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._train_size))
        self.dev: tf.data.Dataset = tf.data.TFRecordDataset('dev.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._dev_size))
        self.test: tf.data.Dataset = tf.data.TFRecordDataset('test.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._test_size))

    class EditDistanceMetric(tf.metrics.Mean):
            def __init__(self, name: str = "edit_distance", dtype = None) -> None:
                super().__init__(name, dtype)

            def update_state(self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor, sample_weight = None) -> None:

                edit_distances = tf.edit_distance(y_pred.to_sparse(), y_true.to_sparse(), normalize=True)
                return super().update_state(edit_distances, sample_weight)
