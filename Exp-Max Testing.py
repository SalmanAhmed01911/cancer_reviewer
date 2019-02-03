#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import py_wsi
import py_wsi.imagepy_toolkit as tk
import matplotlib.pyplot as plt
import scipy.misc
import py_wsi.patch_reader as pr
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator
from glob import glob
from xml.dom import minidom
from shapely.geometry import Polygon, Point
import time
import imageio
import numpy as np
import gc
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.layers import  Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout, UpSampling2D, Input
from keras.models import Sequential, Model
import cv2
import os
from sklearn.manifold import TSNE


# In[2]:


def get_sample_for_testing(path, no_samples):
    filepaths = os.listdir(path)
    no_cancer = []
    cancer = []
    samples = 0
    sample = 0
    while sample < no_samples and samples < len(filepaths):
        paths_to_no_cancer = os.path.join(os.path.join(path, filepaths[samples]), "0")
        paths_to_cancer = os.path.join(os.path.join(path, filepaths[samples]), "1")
        
        patient_samples_no_cancer = os.listdir(paths_to_no_cancer)
        patient_samples_with_cancer = os.listdir(paths_to_cancer)
        
        each_patch = 0
        while each_patch < len(patient_samples_no_cancer) and each_patch < len(patient_samples_with_cancer)  and sample < no_samples:
            no_cancer.append(cv2.resize(cv2.imread(os.path.join(paths_to_no_cancer,patient_samples_no_cancer[each_patch])), (50,50), interpolation=cv2.INTER_CUBIC))
            cancer.append(cv2.resize(cv2.imread(os.path.join(paths_to_cancer,patient_samples_with_cancer[each_patch])), (50,50), interpolation=cv2.INTER_CUBIC))
            sample += 1
            each_patch += 1
        samples += 1
    return no_cancer, cancer


# In[3]:


no_cancer, cancer = get_sample_for_testing("data", 12000)


# In[4]:


no_cancer = np.array(no_cancer)
cancer = np.array(cancer)
no_cancer = no_cancer/255.0
cancer = cancer/255.0
cancer.shape, no_cancer.shape


# In[5]:


data = np.row_stack([cancer, no_cancer])
data.shape


# In[6]:


del no_cancer, cancer
gc.collect()


# In[7]:


y = [1]*12000
y.extend([0]*12000)
len(y)


# In[8]:


y = np.array(y)


# In[9]:


labels = y.copy()


# In[ ]:





# In[10]:


def get_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


# In[11]:


not_converged = True
iter_No = 0
max_iter = 300


# In[12]:


while not_converged and iter_No < max_iter:
    model = get_model()
    model.fit(data, labels, batch_size=128, epochs=2, shuffle=True)
    pred = model.predict_proba(data)
    pred = pred[:,0]
    prevLabels = labels.copy()
    labels[pred>=0.70] = 1
    labels[pred<=0.30] = 0
    print ("###################################")
    print ("###################################")
    print ("###################################")
    print ("###################################")
    print ("ITERATION NO " ,iter_No+1)
    print ("Updated Values ",prevLabels[prevLabels!=labels].shape[0])
    print ("Accuracy", (y[y==labels].shape[0])/float(labels.shape[0]))
    if prevLabels[prevLabels!=labels].shape[0] == 0:
        not_converged = False
    
    temp = pred[pred<0.70]
    temp = temp[temp>0.30]
    print (temp.shape[0], " Should be Dropped")
    temp = pred<0.70
    temp1 = pred>0.30
    temp = temp*temp1
    temp = ~temp
    data = data[temp]
    y = y[temp]
    labels = labels[temp]
    
    print ("Dropped Patches Till Now ", 24000 - data.shape[0])
    iter_No += 1


# In[15]:


print (np.unique(labels, return_counts=True))


# In[16]:


print (np.unique(y, return_counts=True))


# In[ ]:




