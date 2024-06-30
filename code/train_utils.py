import tensorflow as tf
import rasterio
import numpy as np
import os
from keras.callbacks import ModelCheckpoint


class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 paths_list,
                 batch_size,
                 epoch,
                 target_size=(512, 512),
                 shuffle=True,
                 seed=None,
                 w=False,
                 x=False,
                 ):
        
        self.epoch = epoch + 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.paths_list = paths_list
        self.target_size = target_size
        self.n = len(paths_list)
        self.w = w
        self.x = x
        if x:
            self.channels_to_select = [0,1,2,3,4,5,6,7, 9, 10]

    #def on_epoch_end(self):
        #if self.shuffle:
            #pass
    
    def __get_input(self, path, target_size):
        
        # file_name = os.path.basename(path)[0:5]
                
        image = rasterio.open(path).read()
        if len(image.shape)>0:
            image = np.rollaxis(image, 0, 3)
        
        if image.shape[-1] == 1:
            image = image[..., 0]
            
        # else:
        #     pass #image=image[..., np.newaxis]
      

        #image = tf.image.resize(image,(target_size[0], target_size[1])).numpy()
        
        if self.w:
            return (image > 0).astype(float)
        if self.x:
            return image[..., self.channels_to_select]
        return image

    
    def __get_data(self, batch_path):
        # Generates data containing batch_size samples
        X_batch = np.asarray([self.__get_input(image_path, self.target_size) for image_path in batch_path])
        return X_batch
    
    def __getitem__(self, index):
        
        i = (index*self.batch_size) % self.n
        batch_path = self.paths_list[i:i+self.batch_size]
        X = self.__get_data(batch_path)        
        return X
    
    def __len__(self):
        return   self.epoch*self.n // self.batch_size
    

def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        

def refine(ls, black_word='CORRUPTED_'):
    return [s for s in ls if black_word not in s]


class CustomDataGen32(tf.keras.utils.Sequence):
    
    def __init__(self,
                 paths_list,
                 batch_size,
                 epoch,
                 target_size=(512, 512),
                 shuffle=True,
                 seed=None,
                 w=False,
                 x=False,
                 ):
        
        self.epoch = epoch + 1
        self.batch_size = batch_size*4
        self.shuffle = shuffle
        self.seed = seed
        self.paths_list = paths_list
        self.target_size = target_size
        self.n = len(paths_list) * 4
        self.w = w
        self.x = x
        if x:
            self.channels_to_select = [0,1,2,3,4,5,6,7, 9, 10]

    
    #def on_epoch_end(self):
        #if self.shuffle:
            #pass
    
    def __get_input(self, path, target_size):
        
        # file_name = os.path.basename(path)[0:5]
                
        image = rasterio.open(path).read()
        if len(image.shape)>0:
            image = np.rollaxis(image, 0, 3)
        
        if image.shape[-1] == 1:
            image = image[..., 0]
            
        # else:
        #     pass #image=image[..., np.newaxis]
      

        #image = tf.image.resize(image,(target_size[0], target_size[1])).numpy()
        
        if self.w:
            image = image[..., 0]
            return 
        if self.x:
            image = image[..., self.channels_to_select]
            return 
        return [image[0:32, 0:32, ...], image[32:64, 32:64, ...], image[32:64, 0:32, ...], image[0:32, 32:64, ...]]

    
    def __get_data(self, batch_path):
        # Generates data containing batch_size samples
        X_batch = np.hstack([self.__get_input(image_path, self.target_size) for image_path in batch_path])
        return X_batch
    
    def __getitem__(self, index):
        
        i = (index*self.batch_size) % self.n
        batch_path = self.paths_list[i:i+self.batch_size]
        X = self.__get_data(batch_path)        
        return X
    
    def __len__(self):
        return   self.epoch*self.n // self.batch_size


@tf.function
def custom_cosine_similarity(y_true, y_pred):
    y_true_norm = tf.math.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.math.l2_normalize(y_pred, axis=-1)

    return tf.reduce_sum(y_true_norm*y_pred_norm, axis=-1)

@tf.function
def custom_cosine_loss(y_true, y_pred):
   sim = custom_cosine_similarity(y_true, y_pred)
   return 1 - sim


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, __checkpoint_epoch, *args, **kwargs):
        self.__checkpoint_epoch = __checkpoint_epoch
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)


    # redefine the save so it only activates after 100 epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.__checkpoint_epoch: super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)