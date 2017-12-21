import numpy as np
import os

DATASET_ROOT_FOLDER = os.path.abspath('datasets')

class DataLoader:

    train = None
    validation = None
    test = None

    @staticmethod
    def load(train_path=None, validation_path=None, test_path=None):
        if train_path is not None:
            DataLoader.train = DataLoader.load_image_data_with_label_at_end(
                os.path.join(DATASET_ROOT_FOLDER, train_path))
        if validation_path is not None:
            DataLoader.validation = DataLoader.load_image_data_with_label_at_end(
                os.path.join(DATASET_ROOT_FOLDER, validation_path))
        if test_path is not None:
            DataLoader.test = DataLoader.load_image_data_with_label_at_end(os.path.join(DATASET_ROOT_FOLDER, test_path))
        return DataLoader

    @staticmethod
    def get_training_data(mode=None):
        """
        get training data
        :return: dict of (images, labels)
        :rtype: dict
        """
        images = DataLoader.train.images
        labels = DataLoader.train.labels
        if mode is None:
            mode = 1
        if mode == 1:
            images = images[0:1000, :]
            labels = labels[0:1000]
        return {
            'images': images,
            'labels': labels
        }

    @staticmethod
    def get_validation_data(mode=None):
        """
        get validation data
        :return: dict of (images, labels)
        :rtype: dict
        """
        images = DataLoader.validation.images
        labels = DataLoader.validation.labels
        if mode is None:
            mode = 1
        if mode == 1:
            images = images[0:1000, :]
            labels = labels[0:1000]
        return {
            'images': images,
            'labels': labels
        }

    @staticmethod
    def get_test_data(mode=None):
        """
            get test data
            :return: dict of (images, labels)
            :rtype: dict
            """
        images = DataLoader.test.images
        labels = DataLoader.test.labels
        if mode is None:
            mode = 1
        if mode == 1:
            images = images[0:1000, :]
            labels = labels[0:1000]
        return {
            'images': images,
            'labels': labels
        }

    @staticmethod
    def load_image_data_with_label_at_end(path):
        data_with_label = np.load(path)
        images = data_with_label[:, 0:-1]
        labels = data_with_label[:, -1]
        return {
            'images': images,
            'labels': labels
        }

