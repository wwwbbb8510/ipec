import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

DATASET_ROOT_FOLDER = os.path.abspath('datasets')

class DataLoader:

    train = None
    validation = None
    test = None
    mode = None
    partial_dataset = None

    @staticmethod
    def load(train_path=None, validation_path=None, test_path=None, height=28, length=28, train_validation_split_point=10000):
        if train_path is not None:
            DataLoader.train = DataLoader.load_image_data_with_label_at_end(
                os.path.join(DATASET_ROOT_FOLDER, train_path), height=height, length=length)
        if validation_path is not None:
            DataLoader.validation = DataLoader.load_image_data_with_label_at_end(
                os.path.join(DATASET_ROOT_FOLDER, validation_path), height=height, length=length)
        elif train_validation_split_point is not None and train_validation_split_point > 0:
            if DataLoader.mode is None or DataLoader.partial_dataset is not None:
                train_validation_split_point = int(DataLoader.train['images'].shape[0] * 0.8)
            splited_train = {
                'images': DataLoader.train['images'][0:train_validation_split_point, :, :, :],
                'labels': DataLoader.train['labels'][0:train_validation_split_point]
            }
            splited_validation = {
                'images': DataLoader.train['images'][train_validation_split_point:, :, :, :],
                'labels': DataLoader.train['labels'][train_validation_split_point:]
            }
            DataLoader.train = splited_train
            DataLoader.validation = splited_validation
        if test_path is not None:
            DataLoader.test = DataLoader.load_image_data_with_label_at_end(os.path.join(DATASET_ROOT_FOLDER, test_path), height=height, length=length)

        logging.debug('Training data shape:{}'.format(str(DataLoader.train['images'].shape)))
        logging.debug('Validation data shape:{}'.format(str(DataLoader.validation['images'].shape)))
        logging.debug('Test data shape:{}'.format(str(DataLoader.test['images'].shape)))
        return DataLoader

    @staticmethod
    def get_training_data():
        """
        get training data
        :return: dict of (images, labels)
        :rtype: dict
        """
        images = DataLoader.train.images
        labels = DataLoader.train.labels

        return {
            'images': images,
            'labels': labels
        }

    @staticmethod
    def get_validation_data():
        """
        get validation data
        :return: dict of (images, labels)
        :rtype: dict
        """
        images = DataLoader.validation.images
        labels = DataLoader.validation.labels

        return {
            'images': images,
            'labels': labels
        }

    @staticmethod
    def get_test_data():
        """
            get test data
            :return: dict of (images, labels)
            :rtype: dict
            """
        images = DataLoader.test.images
        labels = DataLoader.test.labels

        return {
            'images': images,
            'labels': labels
        }

    @staticmethod
    def load_image_data_with_label_at_end(path, height, length):
        data = np.loadtxt(path)
        if DataLoader.mode is None:
            data = data[0:1000, :]
        elif DataLoader.partial_dataset is not None and DataLoader.partial_dataset > 0 and DataLoader.partial_dataset <1:
            # randomly pick partial dataset
            cut_point = int(data.shape[0] * DataLoader.partial_dataset)
            indices = np.random.permutation(data.shape[0])
            training_idx= indices[:cut_point]
            data = data[training_idx, :]
        images = data[:, 0:-1]
        labels = data[:, -1]
        images = np.reshape(images, [images.shape[0], height, length, 1], order='F')

        return {
            'images': images,
            'labels': labels
        }

