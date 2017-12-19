
from tensorflow.examples.tutorials.mnist import input_data

DEFAULT_MODE = 1 # 1:DEBUG - 1000 rows, 2:ALL

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/")


def get_training_data(mode = None):
    """
    get training data
    :return: dict of (images, labels)
    :rtype: dict
    """
    images = mnist.train.images
    labels = mnist.train.labels
    if mode is None:
        mode = 1
    if mode == 1:
        images = images[0:1000, :]
        labels = labels[0:1000]
    return {
        'images': images,
        'labels': labels
    }


def get_validation_data(mode = None):
    """
    get validation data
    :return: dict of (images, labels)
    :rtype: dict
    """
    images = mnist.validation.images
    labels = mnist.validation.labels
    if mode is None:
        mode = 1
    if mode == 1:
        images = images[0:1000, :]
        labels = labels[0:1000]
    return {
        'images': images,
        'labels': labels
    }


def get_test_data(mode=None):
    """
        get test data
        :return: dict of (images, labels)
        :rtype: dict
        """
    images = mnist.test.images
    labels = mnist.test.labels
    if mode is None:
        mode = 1
    if mode == 1:
        images = images[0:1000, :]
        labels = labels[0:1000]
    return {
        'images': images,
        'labels': labels
    }
