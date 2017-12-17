
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def get_training_data():
    """
    get training data
    :return: dict of (images, labels)
    :rtype: dict
    """
    images = mnist.train.images
    labels = mnist.train.labels
    return {
        'images': images,
        'labels': labels
    }


def get_validation_data():
    """
    get validation data
    :return: dict of (images, labels)
    :rtype: dict
    """
    images = mnist.validation.images
    labels = mnist.validation.labels
    return {
        'images': images,
        'labels': labels
    }


def get_test_data():
    """
        get test data
        :return: dict of (images, labels)
        :rtype: dict
        """
    images = mnist.test.images
    labels = mnist.test.labels
    return {
        'images': images,
        'labels': labels
    }