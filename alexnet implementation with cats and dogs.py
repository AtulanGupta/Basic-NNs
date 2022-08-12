# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 00:44:43 2022

@author: Atulan
"""
# required Func
# 1. Data processing
import numpy as np

import os
import glob #tqdm
from sklearn import preprocessing
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from tensorflow.keras.utils import to_categorical

# Neural Network
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model

# visualization
import matplotlib.pyplot as plt
# %matplotlib inline

# OpenCV
import cv2

#Warnings
import warnings
warnings.filterwarnings("ignore")


# LOAD DATA
testimageDatasetspath = "G:/study/study mart c/Lecture/L-14-AlexNet/dataset/archive/train/train/"
datasetsNamesTest = os.listdir(testimageDatasetspath)

# A Function to resize all images
def imageFeatureExtract(image, size = (227, 227)):
    return cv2.resize(image, size)

Datasetspath = "G:/study/study mart c/Lecture/L-14-AlexNet/dataset/archive/train/train/"
datasetPath = os.path.join(Datasetspath, "*g")
fileRead = glob.glob(datasetPath)                  # loads all individual image from the path



# Preparain Dataset
data = []      # it will store binary data of our resized image
category_or_class = []     # it contain the class name i.e. cat, or dog

for (i, file) in enumerate(fileRead):
    if i > 2000 and i < 24000:
        continue
    if i % 1000 == 0:
        print(i)
    data.append(imageFeatureExtract(cv2.imread(file)))        # generates a data of shape (10001, 227, 227, 3)
    category_or_class.append(file.split(os.path.sep)[-1].split(".")[0])       
    




# Preparing x and y values for our model
data = np.array(data, dtype = np.uint8)    # x - is a array of shape (10001, 227, 227, 3) of binary type, ready for model
labels = np.array(category_or_class)       # y in wordcontaining name dog and cat, need to be converted to binaryg 
print(type(data))
print(type(labels))
np.unique(category_or_class)
labels  = np.array(category_or_class)
label_Encoder = LabelEncoder()




# Data (y value) scalling i.e converting from word to decimal and then finally decimal to binary (using one hot encoder)
#label_Encoder = preprocessing.LabelEncoder
classNames = label_Encoder.fit_transform(labels)
classNames
np.unique(classNames)
oneHotEncoder = OneHotEncoder(sparse = False)
label_EncoderValue = classNames.reshape(len(classNames), 1)
print(label_EncoderValue)
oneHotEncoderValues = oneHotEncoder.fit_transform(label_EncoderValue)

oneHotEncoderValues


# splitting train and test data
(X_train, X_test, Y_train, Y_test) = train_test_split(data, oneHotEncoderValues, test_size=0.2, random_state=42)



# building model
models = Sequential()
#1st Conv2D Layer
models.add(Conv2D(96, kernel_size = (11, 11), strides = (4, 4), 
                 padding = "valid", activation  = 'relu', input_shape = (227, 227, 3)))
models.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2), padding = "valid",
                       data_format = None))



#2nd Conv2D Layer

models.add(Conv2D(256, kernel_size = (5, 5), strides = 1, 
                 padding = "same", activation  = 'relu'))

models.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2), padding = "valid",#"same"
                       data_format = None))




#3rd Conv2D Layer
models.add(Conv2D(384, kernel_size = (3, 3), strides = 1, 
                 padding = "same", activation  = 'relu'))



#4th Conv2D Layer
models.add(Conv2D(384, kernel_size = (3, 3), strides = 1, 
                 padding = "same", activation  = 'relu'))


#5th Conv2D Layer

models.add(Conv2D(256, kernel_size = (3, 3), strides = 1, 
                 padding = "same", activation  = 'relu'))

models.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2), padding = "valid",#"same"
                       data_format = None))


# Flatten Layer
models.add(Flatten())

models.add(Dense(4096, activation = 'relu'))
models.add(Dense(4096, activation = 'relu'))
#models.add(Dense(1000, activation = 'relu'))
models.add(Dense(2, activation = 'softmax'))



models.compile(loss = "categorical_crossentropy",
              optimizer = 'adam',
              metrics = ['accuracy'])

with tf.device('/cpu:0'):
    models.fit(X_train, Y_train,
          epochs = 10,
          validation_data = (X_test, Y_test), 
          verbose = 1)



















 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    































