from keras.models import load_model
import h5py

model = load_model('F:/ingnew3/图像人脸定位/model.h5')
model.load_weights('F:/ingnew3/图像人脸定位/weights.h5')

import os
filelist = []
filenamelist = []
file_dir = 'F:/ingnew3/图像人脸定位/movie_emotion/'
for root, dirs, files in os.walk(file_dir):
    filelist.append(files)
filenamelist=filelist[1:8]

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import glob
def get_one_pred(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)
    test_datagen.fit(x)
    test_data = test_datagen.flow(x)
    preds = model.predict(test_data[0])
    
    return preds

b = []
for i in range(len(filenamelist)):
    for j in range(len(filenamelist[i])):
        a = get_one_pred(file_dir+str(i)+'/'+filenamelist[i][j])
        tmp = a[0].tolist()
        tmp1 = tmp.index(max(tmp))
        b.append(tmp1)

print(b)