import numpy as np
import tensorflow as tf
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
save_model_name = 'tumor_test.h5'
ts_image_list = [f for f in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(os.path.join(TEST_IMAGE_DIR, f))]


from keras.models import load_model
loaded_model = load_model(os.path.join(MODEL_DIR,save_model_name))
loaded_model.summary()

def test_prediction(model):
    
    for file in ts_image_list:
        input_tensor = []
        image_data, image_header = load(os.path.join(TEST_IMAGE_DIR, file))
        
        nib_img = nib.load(os.path.join(TEST_IMAGE_DIR, file))
        w, h, d = image_data.shape
        
        for j in range(int(d/5)):
            input_tensor.append(image_data[:,:,j*5:(j+1)*5])
        
        input_tensor = np.array(input_tensor)
        input_tensor = np.expand_dims(input_tensor, axis=4)
        
        pred_out = []
        for i in range(input_tensor.shape[0]):
            out = model.predict(input_tensor[i:i+1])
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
            pred_out.append(out)
        pred_out = np.array(pred_out)
        
        
        idx = 0
        for lst in pred_out:
            if idx == 0:
                pred_out_file = lst
            else:
                pred_out_file = np.concatenate( (pred_out_file,lst), axis=2)
            idx += 1
        
        #save(pred_out_file, os.path.join(OUPUT_DIR, file))
        nib.save(nib_img, os.path.join(OUPUT_DIR, file))
        print (pred_out_file.shape, file)

test_prediction(loaded_model)