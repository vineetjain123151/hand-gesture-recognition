#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf


# In[3]:


import keras
from keras.preprocessing.image import ImageDataGenerator


# In[8]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_data=train_datagen.flow_from_directory(
        'E:/machine learning/Dataset/training',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[9]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_data= test_datagen.flow_from_directory('E:/machine learning/Dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[10]:


cnn=tf.keras.models.Sequential()


# In[11]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))


# In[12]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[13]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[14]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[15]:


cnn.add(tf.keras.layers.Flatten())


# In[16]:


cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))


# In[17]:


cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))


# In[18]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:





# In[21]:


model=cnn.fit_generator(training_data,
                  steps_per_epoch = 494,
                  epochs = 25,
                  validation_data = test_data,
                  validation_steps = 133)


# In[22]:


from keras.preprocessing import image
test_image = image.load_img('E:/machine learning/Dataset/test_image.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0][0] == 1:
    print('palm')
elif result[0][1] == 1:
    print('I')
elif result[0][2] == 1:
    print('fist')
elif result[0][3] == 1:
    print('fist moved')
elif result[0][4] == 1:
    print('thumb')
elif result[0][5] == 1:
    print('index')
elif result[0][6] == 1:
    print('ok')
elif result[0][7] == 1:
    print('palm moved')
elif result[0][8] == 1:
    print('C')
elif result[0][9] == 1:
    print('down')


# In[3]:





# In[ ]:





# In[55]:





# In[ ]:




