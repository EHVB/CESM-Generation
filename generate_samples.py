from utils.create_data_path import create_data_path
from models.cyclegan import get_cyclegan_model
from data.dataset import Dataset
import keras
from utils.create_data_path import create_data_path_old
from utils.generate_image import generate_images
import matplotlib.pyplot as plt 
import os
import tensorflow as tf


dm_dir = r"./CDD-CESM/Low energy images of CDD-CESM" # Replace with path of digital mammo directory
cesm_dir = r"./CDD-CESM/Subtracted images of CDD-CESM" # Replace with path of contrast mammo directory

dm_paths, cesm_paths = create_data_path_old(dm_dir, cesm_dir)



#DS = Dataset()
#train_ds = DS.create_dataset(dm_paths, cesm_paths)

def normalize(self, dm, cesm):
    dm = (dm / 127.5) - 1.0
    cesm = (cesm / 127.5) - 1.0
    
    return dm, cesm

def pad_to_square(self, image, laterality=None):

    w,h=tf.shape(image)[1],tf.shape(image)[0]
    
    if w > h:
        padded_image = tf.image.pad_to_bounding_box(
            image,
            offset_height=0,  
            offset_width=0,  
            target_height=w,
            target_width=w
        )
    else:
        pad_width = tf.maximum(0, h - tf.shape(image)[1])
        
        ## in case laterality is not available
        
        left_zeros = tf.reduce_sum(tf.cast(image[:,:tf.shape(image)[1]//2] == 0, tf.int32))
        right_zeros = tf.reduce_sum(tf.cast(image[:,tf.shape(image)[1]//2:] == 0, tf.int32))

        if left_zeros < right_zeros:
            pad_width=0
        
        
        # if laterality == "L":
        #     pad_width = 0
            
        padded_image = tf.image.pad_to_bounding_box(
            image,
            offset_height=0,  
            offset_width=pad_width,  
            target_height=h,
            target_width=h
        )
        
    return padded_image 

def random_crop(self, dm, cesm):
    stacked = tf.stack([dm,cesm], axis=0)
    cropped_image = tf.image.random_crop(stacked, size=[2, self.image_size, self.image_size, self.n_channels])
    
    return cropped_image[0], cropped_image[1]

@tf.function    
def load_data(self, dm_path, cesm_path):
    
    # dm_laterality = tf.strings.split(dm_path,"@")[0]
    # dm_image_path = tf.strings.split(dm_path,"@")[1]
    
    dm_image = tf.io.read_file(dm_path)
    dm_image = tf.image.decode_png(dm_image,channels=1)
    
    dm_padded_image = self.pad_to_square(dm_image)
    
    dm_image = tf.cast(dm_padded_image, tf.float32)
    
    #######################################################
    
    # cesm_laterality = tf.strings.split(cesm_path,"@")[0]
    # cesm_image_path = tf.strings.split(cesm_path,"@")[1]
    
    cesm_image = tf.io.read_file(cesm_path)
    cesm_image = tf.image.decode_png(cesm_image,channels=1)
    
    cesm_padded_image = self.pad_to_square(cesm_image)
    
    cesm_image = tf.cast(cesm_padded_image, tf.float32)

    return (dm_image, cesm_image)

@tf.function
def random_jitter(self, dm, cesm):
    dm = tf.image.resize(dm, [286,286])
    dm = (dm / 127.5) - 1.0
    cesm = tf.image.resize(cesm, [286,286])
    cesm = (cesm / 127.5) - 1.0
    
    dm, cesm = self.random_crop(dm, cesm)
    
    if tf.random.uniform(()) > 0.5:
        dm = tf.image.flip_left_right(dm)
        cesm = tf.image.flip_left_right(cesm)
    
    return dm, cesm

dataset = tf.data.Dataset.from_tensor_slices((dm_paths,cesm_paths))

dataset = (
                dataset
                .shuffle(len(dataset))
                .map(load_data, num_parallel_calls = tf.data.AUTOTUNE)
                .map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(1)
                .prefetch(tf.AUTOTUNE)
            )


cyclegan_model = get_cyclegan_model()

cyclegan_model.built = True

cyclegan_model.load_weights("./weights/cyclegan_pretrained.weights.h5") # path to new model weights 

# generate 20 sample images 
generate_images(cyclegan_model.gen_G,dataset,20)