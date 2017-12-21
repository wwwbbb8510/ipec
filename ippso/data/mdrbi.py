from .core import DataLoader
import os

train_path = os.path.join('mnist_background_images', 'mnist_background_images_test.amat')
test_path = os.path.join('mnist_background_images', 'mnist_background_images_train.amat')

loaded_data = DataLoader.load_image_data_with_label_at_end(train_path=train_path, test_path=test_path)