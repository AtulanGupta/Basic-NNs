# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:53:19 2022

@author: Atulan
"""
# func for data processing
import numpy as np
import os          # The OS module in Python provides functions for interacting with the operating system like when we want to work with directory.
import glob        # to read multiple images from a specific folder (tqdm, pillow)
from sklearn import preprocessing 
from sklearn.preprocessing import *         # using * we can call all functions of that module
from sklearn.model_selection import train_test_split     # jabotio procesing related sob e model_selection theke pabo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  
''' LevelEncoder Vs OneHotEncoder- 
 Level encoder- for 10 classes data will be categorized between 0 and 10.  
 One hot encoder- all data will be categorized with either 0 or 1 
 that is values may be 0,0.1,0.005,....1. but minimum must be 0 and max must be 1.
 Technique- first do Level encoder at first layer, then use the o/p of Level encoder to one hot encoder.
'''
 
# binarize- if we have binary classification then we can use only label_binarize, so that it will serve the purpose of both 1hot and Level Encoder.
# Func for Neural Network 
import tensorflow as tf         # for imgae processin we need tensopr flow
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten, MaxPooling2D    # from network architecture we can say that we need
                                                                # conv layer, dense layer, 
from tensorflow.keras.optimizers import SGD   # SGD- Stochastic gradient descent 
from tensorflow.keras.models import Sequential, load_model

# visualization
import matplotlib.pyplot as plt
%matplotlib inline

# OpenCV
import cv2         # as I am doing image processing I need this

#Warnings
import warnings
warnings.filterwarnings("ignore")




# Load our raw data
testimageDatasetspath = "G:/study/study mart c/Lecture/L-14-AlexNet/dataset/archive/train/train/"    # we have loaded our images  

datasetsNamesTest = os.listdir(testimageDatasetspath) # it creates a list containing all the image names. 
                                                      # It is used to get the list of all files and directories in the specified directory.
   # 0s.listdir- returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order. 
   # It does not include the special entries '.' and '..' even if they are present in the directory.
#a = list(np.random.randint(0, 100, size = 20))  # it will take 20 random int values.
#print(a)            

print( np.unique(datasetsNamesTest))    # select a random value

print(datasetsNamesTest.index("dog.159.jpg")) # check index number of a specific data    

print( datasetsNamesTest[4471])        # checkin/retreivin  data                

image_width = 150
image_height = 150

#def imageFeatureExtract(image, size = (32, 32)):
#    return cv2.resize(image, size).flatten()   # these 2 lines generate an error

def imageFeatureExtract(image, size = (28, 28)):
    img = cv2.resize(image, size)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # (28,28)
    gray = np.expand_dims(gray, 2)#(28, 28, 1)
    return gray



# Checking defined func with an image 
testimageDatasetspath = cv2.imread("G:/study/study mart c/Lecture/L-14-AlexNet/dataset/archive/train/train/cat.100.jpg")
imageFeatureExtract(testimageDatasetspath)
testimageDatasetspath = "G:/study/study mart c/Lecture/L-14-AlexNet/dataset/archive/train/train/"           # load the file path



# make a list containg all raw data source
datasetPath = os.path.join(testimageDatasetspath, "*g")   # join another path with previous path 
fileRead = glob.glob(datasetPath)        




# preparing data for our NN
data = []                                                 
category_or_class = []                                    

for (i, file) in enumerate(fileRead):
    image = cv2.imread(file)     # imread converts image into binary data set
    feature = imageFeatureExtract(image)
    data.append(feature)        # list containing data for our x set in binary format 
    classNames = file.split(os.path.sep)[-1].split(".")[0] 
    category_or_class.append(classNames)  #  # list containing data for our x set in word format, need to be converted in binary format

np.unique(category_or_class)

# for i, j in enumerate(a):        enumerate dile i- index number, j- corresponding value of that index, i.e enumerate shows bothh index and value
                                  # enumerate na diye range dile i- list er value gulo dekhato sudhu.
#     print(i, j)

#cv2.imread("Datasets/train/train/cat.100.jpg")  # cv2.imread diye amra image ke array te convert kore read kori 

# #"Datasets/train/train/cat.100.jpg".split()       # creates a set of string
# #"Datasets/train/train/cat.100.jpg".split("/")    # creates a set of list with 1 dimension that is of single array, split data were it finds a slash(/)
                                                    # creates a set of list with 1 dimension that is of single array, split data were it finds a dot(.) 
# "Datasets/train/train/cat.100.jpg".split("/")[3].split(".")[0]      # creates a single string

print(len(data))
 
print(len(category_or_class))

print(type(category_or_class))




# Scaling our data in y set(category_or_class)
labels  = np.array(category_or_class)
label_Encoder = LabelEncoder()
#label_Encoder = preprocessing.LabelEncoder
classNames = label_Encoder.fit_transform(labels) # converted y set - decimal format

np.unique(classNames)

classNames.shape

classNames[0]

oneHotEncoder = OneHotEncoder(sparse = False)
label_EncoderValue = classNames.reshape(len(classNames), 1)
oneHotEncoderValues = oneHotEncoder.fit_transform(label_EncoderValue) # converted y set - binary format

#help(OneHotEncoder() )

# print(oneHotEncoderValues)

# data = ["Cats", "Dogs", "Birds"]
# np.array([[1., 0., 0],
#        [0., 1., 0.],
#        [0., 0., 1.]])

# Datasets Scalling
data = np.array(data)/255.0




# splitting data set
(X_train, X_test, Y_train, Y_test) = train_test_split(data, oneHotEncoderValues, test_size=0.2, random_state=42)



# building NN
model = Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels(4096) is much larger than that in LeNet

#1st Conv2D Layer
model = Sequential() # Empty 

# 1st layer
model.add(Conv2D(6,kernel_size = (5, 5), strides = (1, 1), activation = "tanh", input_shape = (28, 28, 1), padding = "same"))
model.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))

# 2nd Layer
model.add(Conv2D(16,kernel_size = (5, 5), strides = (1, 1), activation = "tanh", padding = "valid"))
model.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))

# Flatten Layer
model.add(Flatten())

# Output Layer
model.add(Dense(120, activation = "tanh"))
model.add(Dense(84, activation = "tanh"))
model.add(Dense(2, activation = "softmax"))

model.compile(optimizer = "adam",
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])


model.fit(X_train, Y_train,
          epochs = 10,
          validation_data = (X_test, Y_test), 
          verbose = 1)
