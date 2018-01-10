from .core import DataLoader
import os

train_path = os.path.join('mnist_background_images', 'mnist_background_images_train.amat')
test_path = os.path.join('mnist_background_images', 'mnist_background_images_test.amat')

loaded_data = DataLoader.load(train_path=train_path, test_path=test_path)