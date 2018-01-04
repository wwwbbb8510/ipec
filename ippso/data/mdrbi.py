from .core import DataLoader
import os

train_path = os.path.join('mnist_rotation_back_image_new', 'mnist_all_background_images_rotation_normalized_train_valid.amat')
test_path = os.path.join('mnist_rotation_back_image_new', 'mnist_all_background_images_rotation_normalized_test.amat')

loaded_data = DataLoader.load(train_path=train_path, test_path=test_path)