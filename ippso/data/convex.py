from .core import DataLoader
import os

train_path = os.path.join('convex', '50k', 'convex_test.amat')
test_path = os.path.join('convex', 'convex_train.amat')

loaded_data = DataLoader.load(train_path=train_path, test_path=test_path, train_validation_split_point=6000)