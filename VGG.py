# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:31:31 2022

@author: Atulan
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Activation, Dropout, MaxPooling2D
from tensorflow.keras import backend as k


from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# %matplotlib inline

trainDatasets = "G:/study/study mart c/Lecture/L-16- Vgg/dogs-vs-cats/test/test/"
testDatasets = "G:/study/study mart c/Lecture/L-16- Vgg/dogs-vs-cats/train/train/"

datasetsClassLabels = os.listdir(trainDatasets)
datasetsClassLabels

def class_Names(classList):
     classNames = []
     for i in classList:
         names = i.split('.')[0]
         classNames.append(names)
     return list(np.unique(classNames))

classNames = class_Names(datasetsClassLabels)

classNames
'''
catImage = cv2.imread("G:/study/study mart c/Lecture/L-09-11-LetNet/datasets/train/train/dog.97.jpg")
plt.imshow(catImage)
plt.show()'''
'''
catImage = cv2.imread("Datasets/training_set/dogs/dog.100.jpg")
plt.imshow(catImage)
plt.show()'''

trainDataGenerator = ImageDataGenerator(zoom_range = 0.20,
                                       width_shift_range = 0.20, 
                                       height_shift_range = 0.20, 
                                       shear_range = 0.20)

testDataGenerator = ImageDataGenerator()
trainDataGenerator = ImageDataGenerator()
trainDataGenerator = trainDataGenerator.flow_from_directory("G:/study/study mart c/Lecture/L-16- Vgg/dogs-vs-cats/train/train",
                                                           target_size = (224, 224),
                                                           batch_size = 32, 
                                                           shuffle = True, 
                                                           class_mode = "binary")

testDataGenerator = testDataGenerator.flow_from_directory("G:/study/study mart c/Lecture/L-16- Vgg/dogs-vs-cats/test/test",
                                                          target_size = (224, 224),
                                                           batch_size = 32, 
                                                           shuffle = False,
                                                           class_mode = "binary")
'''
model = Sequential()

# 1st Layer
model.add(Conv2D(input_shape = (224, 224, 3), filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# 2nd Layer
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#3rd Layer
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#4th Layer
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#5th Layer
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'vgg16'))

model.add(Flatten(name = "flatten"))


model.add(Dense(256, activation = 'relu', name = 'fcN1'))
model.add(Dense(128, activation = 'relu', name = 'fcN2'))
model.add(Dense(1, activation = "sigmoid", name = 'output'))

model.summary()


vgg_models = Model(inputs = model.input, outputs = model.get_layer('vgg16').output)

vgg_models.load_weights("G:/study/study mart c/Lecture/L-16- Vgg/vgg16_weights_tf_dim_ordering_tf_kernels_notop-2.h5")
                        

for modelLayers in vgg_models.layers:
    #print(modelLayers)
    modelLayers.trainable = False
# 1. online learning - dynamic learning
# 2. offline learning - Static Learning 
#  Reinforcement Learning - RL 

for model_layer in model.layers:
    print("Layer: ", model_layer, "\nTrainable Layer: ", model_layer.trainable, "\n")
    
optimizer = SGD(learning_rate = 1e-4, momentum = 0.9)

model.compile(loss = "binary_crossentropy",
             optimizer = optimizer,
             metrics = ["accuracy"])

# 1. Early Stoping
# 2. Regularization
# 3. Dropout

earlyStoping = EarlyStopping(monitor = 'val_accuracy',
                            mode = "max", 
                            verbose = 1, 
                            patience = 2)'''


# # 1. value loss
# # 2. value accuracy 
#     i. Max
#     ii. min
# # 3. accuracy
# # 4. loss

'''model.fit(trainDataGenerator,
          validation_data = testDataGenerator,
          epochs = 5,
          verbose = 1,
          callbacks = earlyStoping)'''

'''
a = [[1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5]]

a.append(10)
'''
'''model.fit(trainDataGenerator,
          validation_data = testDataGenerator,
          epochs = 5,
          verbose = 1,
          callbacks = earlyStoping)'''








































































































































