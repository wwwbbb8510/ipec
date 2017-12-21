from .core import DataLoader
import os

train_path = os.path.join('mnist', 'mnist_train.amat')
test_path = os.path.join('mnist', 'mnist_test.amat')

DataLoader.load_image_data_with_label_at_end(train_path=train_path, test_path=test_path)
loaded_data = DataLoader