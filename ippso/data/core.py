import numpy as np
import os

DATASET_ROOT_FOLDER = os.path.abspath('datasets')

class DataLoader:

    train = None
    validation = None
    test = None
    mode = None

    @staticmethod
    def load(train_path=None, validation_path=None, test_path=None, height=28, length=28, mode=None, train_validation_split_point=10000):
        DataLoader.mode = mode
        if train_path is not None:
            DataLoader.train = DataLoader.load_image_data_with_label_at_end(
                os.path.join(DATASET_ROOT_FOLDER, train_path), height=height, length=length)
        if validation_path is not None:
            DataLoader.validation = DataLoader.load_image_data_with_label_at_end(
                os.path.join(DATASET_ROOT_FOLDER, validation_path), height=height, length=length)
        elif train_validation_split_point is not None and train_validation_split_point > 0:
            if DataLoader.mode is None:
                train_validation_split_point = int(DataLoader.train['images'].shape[0] * 0.8)
            splited_train = {
                'images': DataLoader.train['images'][0:train_validation_split_point, :, :, :],
                'labels': DataLoader.train['images'][0:train_validation_split_point]
            }
            splited_validation = {
                'images': DataLoader.train['images'][train_validation_split_point:, :, :, :],
                'labels': DataLoader.train['images'][train_validation_split_point:]
            }
            DataLoader.train = splited_train
            DataLoader.validation = splited_validation
        if test_path is not None:
            DataLoader.test = DataLoader.load_image_data_with_label_at_end(os.path.join(DATASET_ROOT_FOLDER, test_path), height=height, length=length)
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
        images = data[:, 0:-1]
        labels = data[:, -1]
        images = np.reshape(images, [images.shape[0], height, length, 1], order='F')

        return {
            'images': images,
            'labels': labels
        }

