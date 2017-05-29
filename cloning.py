
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Flatten, Layer, Dense, Convolution2D, Activation, Dropout
from keras.layers import Cropping2D, MaxPooling2D
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.layers.core import Lambda

# In[2]:

center_images = []
steering = []


# In[3]:

#load center images
with open('./data/driving_log.csv', 'r') as drivinglogfile:
    next(drivinglogfile, None)
    reader = csv.reader(drivinglogfile)
    for line in reader:
	    #print(line[0].split("/")[-1])
	    for i in range(3):
		    img = cv2.imread("./data/IMG/"+line[i].split("/")[-1])
		    center_images.append(img)
		    if ( i==1):
			    steering.append(float(line[3]) + 0.18)
		    if ( i==2 ):
			    steering.append(float(line[3]) - 0.18)
		    if ( i==0):
			    steering.append(line[3])
	    #print(line[3])



print(len(center_images))
print(len(steering))



#convert training data to numpy arrays
X_train = np.array(center_images)
y_train = np.array(steering)
print(keras.__version__)
print(X_train.shape)
input_shape = X_train.shape[1:]


# In[6]:

#create the DL model
model = Sequential()
model.add( Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape) )
#60,30 below worked for the most part
model.add(Cropping2D(cropping=((70,30), (0,0))))
model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode="same") )
model.add( Activation('relu'))

model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add( Activation('relu'))

model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add( Activation('relu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add( Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))
model.add(Flatten())

model.add(Dense(16))
model.add( Activation('relu'))
model.add(Dense(16))
model.add( Activation('relu'))
#model.add(Dense(20))
#model.add( Activation('relu'))
#model.add(Dropout(.25))

model.add(Dense(1))

#compile the model
model.compile( loss='mse', optimizer='adam')
#fit data
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')

# In[ ]:



