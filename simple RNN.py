# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 09:12:12 2022

@author: Atulan
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, SimpleRNN, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

from termcolor import colored

allChars = "90348756921+"

print(len(allChars))





# preparing Data for RNN
# 1. CTI
# 2. ITC
char_to_index = dict((c, i) for i, c in enumerate(allChars))
index_to_char = dict((i, c) for i, c in enumerate(allChars))
char_to_index

def numberGenerate():
    firstNumber = np.random.randint(3, 10) # 1 value between 3 and 10
    SecNumber = np.random.randint(3, 10)
    newValue = str(firstNumber) + "+"+str(SecNumber)
    finalValue = str(firstNumber+SecNumber)
    return newValue, finalValue

numberGenerate()

# model = Sequential()
# model.add(Conv2D(128, input_shape = (128, 128))  = 10 
# model.add(Maxpooling2D(pool_size = (2, 2)))) = 7

# model.add(Conv2D(128, activation = "relu"))
# model.add(Maxpooling2D(pool_size = (2, 2))))

model = Sequential([
    
    SimpleRNN(128, input_shape = (None, len(allChars))),
    RepeatVector(3),
    
    
    SimpleRNN(128, return_sequences = True),
    TimeDistributed(Dense(len(allChars), activation = "softmax"))])

model.compile(loss = "categorical_crossentropy",
             optimizer = "adam", 
             metrics = ["accuracy"])

model.summary()

def vector(newvalue, finalvalue):
    
    x = np.zeros((3, len(allChars))) # create a (3,10) array
    y = np.zeros((3, len(allChars))) 
    
    difference_x = 3 - len(newvalue)  # newvalue= 5+4, len(newvalue) = 3, hence difference_x=0
    difference_y = 3 - len(finalvalue) # finalvalue= 9, len(finalvalue)=1, hence difference_y=2
    
    
    for i, c in enumerate(newvalue): # 5+4
        x[difference_x + i, char_to_index[c]] = 1
        
    for i in range(difference_x):
        x[i, difference_x["0"]] = 1
    
    
    for i, c in enumerate(finalvalue):
        y[difference_y + i, char_to_index[c]] = 1
        
    for i in range(difference_y):
        y[i, char_to_index["0"]] = 1  
    
    return x, y

newValue, finalValue = numberGenerate()

print("New Value: ", newValue)
print("Final Value: ", finalValue)

x, y = vector(newValue, finalValue)

print("X Value: ", x)
print("\n")
print("Y Value: ", y)

def ReturnVector(value):
    data = [index_to_char[np.argmax(vector)] for i, vector in enumerate(value)] # take vector, take the max value from that and convert it to char
    return "".join(data)

ReturnVector(x)

ReturnVector(y)

def trainData_and_TestData(num_samples = 100000):
    
    x_trainData = np.zeros((num_samples, 3, len(allChars)))
    y_trainData = np.zeros((num_samples, 3, len(allChars)))
    
    
    for i in range(num_samples):
        newValue, finalValue = numberGenerate()
        x, y = vector(newValue, finalValue)
        
        x_trainData[i] = x
        y_trainData[i] = y
        
    return x_trainData, y_trainData

x_trainData, y_trainData = trainData_and_TestData()

#print("X Train Data: \n", x_trainData)

x_trainData.shape # (10000,3,12)-(num_samples, repetition, length of character)

ReturnVector(y_trainData[0])




# fit our NN
Call_backs =  LambdaCallback(
    on_epoch_end = lambda newValue, finalValue: print("{:.2f}".format(finalValue["accuracy"]), end = "|"))

earlyStop = EarlyStopping(monitor = "val_loss", patience=10)

model.fit(x_trainData, 
         y_trainData,
         epochs=10,
         validation_split=0.2, 
         verbose = False, 
         callbacks = [Call_backs, earlyStop])

















































































