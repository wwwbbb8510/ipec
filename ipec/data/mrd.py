from .core import DataLoader
import os

train_path = os.path.join('mnist_rotation_new', 'mnist_all_rotation_normalized_float_train_valid.amat')
test_path = os.path.join('mnist_rotation_new', 'mnist_all_rotation_normalized_float_test.amat')

loaded_data = DataLoader.load(train_path=train_path, test_path=test_path)