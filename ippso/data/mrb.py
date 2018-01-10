from .core import DataLoader
import os

train_path = os.path.join('mnist_background_random', 'mnist_background_random_train.amat')
test_path = os.path.join('mnist_background_random', 'mnist_background_random_test.amat')

loaded_data = DataLoader.load(train_path=train_path, test_path=test_path)