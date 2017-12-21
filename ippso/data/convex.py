from .core import DataLoader
import os

train_path = os.path.join('convex', '50k', 'convex_test.amat')
test_path = os.path.join('convex', 'convex_train.amat')

loaded_data = DataLoader.load_image_data_with_label_at_end(train_path=train_path, test_path=test_path)