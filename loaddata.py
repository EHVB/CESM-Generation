from utils.create_data_path import create_data_path
from models.cyclegan import get_cyclegan_model
from data.dataset import Dataset
import keras
from utils.create_data_path import create_data_path_old
from utils.generate_image import generate_images
import matplotlib.pyplot as plt 
import os
import tensorflow as tf

import tensorflow_datasets as tfds

print(tfds.__version__)



dm_dir = r"./CDD-CESM/Low energy images of CDD-CESM" # Replace with path of digital mammo directory
cesm_dir = r"./CDD-CESM/Subtracted images of CDD-CESM" # Replace with path of contrast mammo directory

dm_paths, cesm_paths = create_data_path_old(dm_dir, cesm_dir)



DS = Dataset()
train_ds = DS.create_dataset(dm_paths, cesm_paths)

cyclegan_model = get_cyclegan_model()

cyclegan_model.built = True

cyclegan_model.load_weights("./weights/cyclegan_pretrained.weights.h5") # path to new model weights 
print(tf.__version__)
