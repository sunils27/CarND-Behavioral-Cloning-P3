
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Flatten, Layer, Dense, Conv2D
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras


# In[2]:

center_images = []
left_images = []
right_images = []
steering = []


# In[3]:

#load center images
with open('./data/driving_log.csv', 'r') as drivinglogfile:
    next(drivinglogfile, None)
    reader = csv.reader(drivinglogfile)
    for line in reader:
        #print(line[0].split("/")[-1])
        img = cv2.imread("./data/IMG/"+line[0].split("/")[-1])
        center_images.append(img)
        #left_images.append(cv2.imread("./data-mysim/IMG/"+line[1].split("/")[-1]))
        #right_images.append(cv2.imread("./data-mysim/IMG/"+line[2].split("/")[-1]))
        steering.append(line[3])


# In[4]:

print(len(center_images))
print(len(left_images))
print(len(right_images))
print(len(steering))


# In[5]:

#convert training data to numpy arrays
X_train = np.array(center_images)
y_train = np.array(steering)
print(keras.__version__)
print(X_train.shape)


# In[6]:

#create the DL model
model = Sequential()
#model.add( Flatten(input_shape=(160,320,3)) )
model.add( Conv2D(32,3,3, subsample=(4,4), border_mode='valid', input_shape=(160,320,3)  ))
model.add(Dense(1))

#compile the model
model.compile( loss='mse', optimizer='adam')
#fit data
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
# In[ ]:



