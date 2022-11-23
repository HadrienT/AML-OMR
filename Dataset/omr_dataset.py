import tensorflow as tf
import matplotlib.pyplot as plt

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
        self._test_size: int =  7307

        self.train: tf.data.Dataset = tf.data.TFRecordDataset('train.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._train_size))
        self.dev: tf.data.Dataset = tf.data.TFRecordDataset('dev.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._dev_size))
        self.test: tf.data.Dataset = tf.data.TFRecordDataset('test.tfrecord').map(OMRDataset.parse).apply(
            tf.data.experimental.assert_cardinality(self._test_size))

if __name__ == "__main__":
    omr = OMRDataset()
    imgs = []
    labels = []
    for example in omr.train.take(5):
        imgs.append(example["image"])
        labels.append(example["label"])

    _, axs = plt.subplots(5, 1, figsize=(12, 12))
    axs = axs.flatten()

    for img, title, ax in zip(imgs, labels, axs):
        ax.axis('off')
        ax.imshow(img, cmap='afmhot')
        ax.set_title(' '.join([symbol for symbol in title.numpy()]))
    plt.show()