# -*- coding: utf-8 -*-
"""
Created on Tue May 31 01:18:08 2022

@author: Atulan
"""
# func for data processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid    #
import matplotlib.image as matimage   #
import cv2                            # it creates magic function
%matplotlib inline
# ma = []
# for i in range(2550):
#     a = 0.2*np.random.uniform()
#     b = a*255
#     ma.append(b)
    
# print(np.max(ma))
# a = ["Red", "Green", "Blue"]
# a[2]
# for i in range(11):
#     angel = np.random.uniform(30)  - 30/2
#     print(angel)


# changing image brigtness
def brightness(imageNamefilePath):
    imageData = cv2.cvtColor(imageNamefilePath, cv2.COLOR_RGB2HSV)   
    # cv2- color conversation 
    # HSV- hue saturation or separation value. er maddhome amra tinta color er jonno 3 ta vinno matrix pabo.
    Randombright = 0.9*np.random.uniform() # uniform distribution hocche
    imageData[:, :,2] = imageData[:, :,2]* Randombright
    imageData  = cv2.cvtColor(imageData, cv2.COLOR_HSV2RGB)
    return imageData
                                                                           
5* 10 * np.random.uniform() - 10/2


def imageTransformation(imageNamefilePath, angel_range_change, shear_range_change, translation_range):
    
    
    
    # Image Angel Rotation Calculation
    angel_rotation = np.random.uniform(angel_range_change) - angel_range_change / 2
    image_hieght, image_width, color_ch = imageNamefilePath.shape
    image_metrixCalculation = cv2.getRotationMatrix2D((image_width/2, image_hieght/2), angel_rotation, 1) # 1- color channel
    
    
    #Image Translation - chane image feature or other required value
    axisX = translation_range* np.random.uniform() - translation_range/2
    axisY = translation_range* np.random.uniform() - translation_range/2 
    TranslatationConcatenateOrMerge = np.float32([[1, 0, axisX], [0, 1, axisY]])
    
    
    #Image Sheare range Calculation
    shear_range_changeValue = np.float32([[5, 5], [20, 5], [5, 20]])
    
    axisPoint1 = 5+ shear_range_change * np.random.uniform() - shear_range_change/2 # amra ekhane 5 ar 20 ei 2 ta different value niye 
    axisPoint2 = 20+ shear_range_change * np.random.uniform() - shear_range_change/2# korchi, tai ei duitar jonno calculation korchi. 
                                                                                    # aro unique thakle aro point er calculation korte hobe.    
    Points = np.float32([[axisPoint1, 5], [axisPoint2, axisPoint1], [5, axisPoint2]])
    
    shearRangeoncatenateOrMerge  = cv2.getAffineTransform(shear_range_changeValue, Points) # AffineTransform- mainly parrallel change kore. mane
                                                                                           # sathe sathe change kore dey.
    
 # concatenate- pack kora, warp- unpack kora   
    image = cv2.warpAffine(imageNamefilePath, image_metrixCalculation, (image_width, image_hieght)) # merge ; warp- unpacking, twist
    image = cv2.warpAffine(image, TranslatationConcatenateOrMerge, (image_width, image_hieght))
    image = cv2.warpAffine(image, shearRangeoncatenateOrMerge, (image_width, image_hieght))
    
    image = brightness(image) # ekhane brightness child func hisabe kaj korbe
    
    return image





# parameter setting for image show like figsize, grid scale settin, rid space etc
plt.figure(figsize=(10, 8))     
imagefilePath = matimage.imread("G:/study/study mart c/image/Cat.jpg")

plt.imshow(imagefilePath)

plt.axis('off')
plt.show()
gridValue1 = grid.GridSpec(10, 10)
gridValue1.update(wspace = 0.01, hspace = 0.02) #w- width, h- height
plt.figure(figsize=(20,18))



# num of image that i want to replicate

for i in range(100):
    ax1 = plt.subplot(gridValue1[i]) # subplot- ekta axis e multiple value dekhano jay 
                                     # subplots- multiple axis e multiple image read hobe
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')   # aspect- ratio i.e. image er pixel ratio equal thakbe
    image = imageTransformation(imagefilePath, 20, 10, 5)
    
    plt.subplot(10, 10, i+1)  # range 100 chilo, tai 10 X 10 akare likhchi
    plt.imshow(image)
    plt.axis('off')
    
    #.save("../ImageAugmentation/Images/"+i+".png")
matimage.imsave("ImageAugmentation/"+str(i+1)+".png", image)
    
plt.show() '''

# "ImageAugmentation/"


 