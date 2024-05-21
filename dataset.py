import tensorflow as tf

class Dataset():
    def __init__(self, image_size=256, batch_size=1, n_channels=1):
        
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.AUTOTUNE = tf.data.AUTOTUNE
    
    
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
        
        print(tf.shape(dm_image)[0])
        
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
    
    def create_dataset(self, X, y, train=True):
    
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        
        if train:
            dataset = (
                dataset
                .shuffle(len(dataset))
                .map(self.load_data, num_parallel_calls = self.AUTOTUNE)
                .map(self.random_jitter, num_parallel_calls=self.AUTOTUNE)
                .batch(self.batch_size)
                .prefetch(self.AUTOTUNE)
            )
        else:
            dataset = (
                dataset
                .map(self.load_data, num_parallel_calls = self.AUTOTUNE)
                .map(self.normalize, num_parallel_calls = self.AUTOTUNE)
                .batch(self.batch_size)
                .prefetch(self.AUTOTUNE)
            )
        
        return dataset