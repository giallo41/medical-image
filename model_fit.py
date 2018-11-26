
import numpy as np
import tensorflow as tf
import random
import os

import keras
import keras.backend as K

import nibabel as nib
from medpy.io import load
from medpy.io import save


TRAIN_IMAGE_DIR = '/data/train/image'
TRAIN_LABEL_DIR = '/data/train/label'
TEST_IMAGE_DIR = '/data/test'
MODEL_DIR = '/data/model'
OUPUT_DIR = '/data/output'


tr_image_list = [f for f in os.listdir(TRAIN_IMAGE_DIR) if os.path.isfile(os.path.join(TRAIN_IMAGE_DIR, f))]
tr_label_list = [f for f in os.listdir(TRAIN_LABEL_DIR) if os.path.isfile(os.path.join(TRAIN_LABEL_DIR, f))]
ts_image_list = [f for f in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(os.path.join(TEST_IMAGE_DIR, f))]


def batch_data_load(batch_size = 4):
    
    origin_img_shape = []
    x_rtn = []
    y_rtn = []
    
    random_idx = random.sample(range(0, len(tr_label_list)), batch_size)

    # For Testing 
    random_idx = [i for i in range(165,175)]
    
    for i in range(len(random_idx)):
        
#        print (tr_image_list[random_idx[i]*2])
#        print (tr_label_list[random_idx[i]])
        
        image_data, image_header = load(os.path.join(TRAIN_IMAGE_DIR, tr_image_list[random_idx[i]*2]))
        label_data, label_header = load(os.path.join(TRAIN_LABEL_DIR, tr_label_list[random_idx[i]]))
        
        origin_img_shape.append(image_data.shape)
        
        print (image_data.shape, label_data.shape)
        
        w, h, d = image_data.shape
        
        for j in range(int(d/5)):
            x_rtn.append(image_data[:,:,j*5:(j+1)*5])
            y_rtn.append(label_data[:,:,j*5:(j+1)*5])
       # x_rtn.append(image_data)
       # y_rtn.append(label_data)
        
    return np.array(x_rtn), np.array(y_rtn), origin_img_shape #x_rtn, y_rtn, origin_img_shape
        #np.array(x_rtn), np.array(y_rtn), origin_img_shape#np.expand_dims(np.array(x_rtn),axis=4), np.expand_dims(np.array(y_rtn), axis=4)
        

img_size = 8

X_data, y, ori_img_shape = batch_data_load(batch_size = img_size)
print (X_data.shape, y.shape)

X_data = np.expand_dims(X_data, axis=4)
y = np.expand_dims(y, axis=4)
print (X_data.shape, y.shape)
tr_size = int(len(X_data)*0.8)

X_train = X_data[:tr_size]
y_train = y[:tr_size]
X_valid = X_data[tr_size:]
y_valid = y[tr_size:]




from keras import layers
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, Flatten, Reshape, Dropout, Conv3DTranspose
from keras.layers import Concatenate, BatchNormalization, Add
from keras.models import Model, Sequential
from keras.layers import InputLayer
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model




K.clear_session()
def make_model(image_data):
    
    im_h,im_w,im_d, im_c = image_data[0].shape
    print (image_data[0].shape)
    
    input_data = Input(shape=(image_data[0].shape))
    
    x = Conv3D(256, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(input_data)
    x = layers.Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(x)
    x = layers.Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(x)
    x = layers.Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(x)
    x = layers.Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(1, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(x)
    x = layers.Activation('relu')(x)
    x = BatchNormalization()(x)
    #x = MaxPooling3D(pool_size=(2,2,2))(x)
    print (x.shape)
    
    output = x

    model = Model(inputs=input_data, outputs=output)
    
    return model

with tf.device("/gpu:0"):
    model = make_model(X_data)
print (model.summary())


from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

from keras.callbacks import Callback

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr) # K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('LR: {:.6f}'.format(lr))

#tb_hist = keras.callbacks.TensorBoard(log_dir=MODEL_DIR, histogram_freq=0, write_graph=True, write_images=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=100, verbose=0, mode='min')


import time
start_time = time.time() 
EPOCHS = 2
LRATE = 0.001

model.compile(loss=['mean_squared_error'], optimizer=Adam(lr=LRATE,  decay=0.01), metrics=['mean_squared_error'])
history = model.fit(X_train , y_train,
                  batch_size=8,
                  epochs=EPOCHS,
                  callbacks=[early_stopping],# SGDLearningRateTracker()],
                  shuffle = True,
                  verbose=2
                  ,validation_data=(X_valid, y_valid)
                 )

end_time = time.time()
print("--- Train Time : %0.2f hour  ---" %(  (end_time - start_time)/3600  ))

save_model_name = 'tumor_test.h5'
model.save(os.path.join(MODEL_DIR,save_model_name))
