#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import keras
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import load_img, img_to_array, array_to_img
import os
from sklearn.model_selection import train_test_split
import sys
import json
import dvc.api


import warnings
warnings.filterwarnings("ignore")


# In[2]:


path_train = "./csv_storage/train.csv"
path_valid = "./csv_storage/valid.csv"
path_test = "./csv_storage/test.csv"
path_scores = "./scores.json"


# In[3]:


train = pd.DataFrame(columns=["images_src", "mark"])
valid = pd.DataFrame(columns=["images_src", "mark"])
test = pd.DataFrame(columns=["images_src", "mark"])


# In[4]:


train = pd.read_csv(path_train, index_col=False)
test = pd.read_csv(path_valid, index_col=False)
valid = pd.read_csv(path_test, index_col=False)


# In[5]:


def frame_to_serie(frame: pd.DataFrame):
    data = []
    marks = []
    for _, info in frame.iterrows():
        raw_img = load_img(info[0], target_size=(192, 192))
        data.append(img_to_array(raw_img))
        marks.append(info[1])
    
    return data, marks


# In[6]:


def conf_matx(x_test, y_test, model):
    """Построение матрицы ошибок"""
    y_preds = []
    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        y_preds.append(np.argmax(y_pred[i]))
    
    y_test_t = []
    for row in y_test:
        for i in range(0, 2):
            if row[i] == 1:
                y_test_t.append(i)
                

    fig, ax = plt.subplots(figsize=(8,8)) 

    matrix = confusion_matrix(y_test_t, y_preds)
    return heatmap(matrix, annot=True, cmap="Blues", square=True, annot_kws={"fontsize":8}, fmt="g")


# In[7]:


x_train, y_train = frame_to_serie(train)
x_valid, y_valid = frame_to_serie(valid)
x_test, y_test = frame_to_serie(test)


# In[8]:


x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)


# In[15]:


from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import MobileNet
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras import layers
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from keras.metrics import Precision, Recall, AUC
import copy


# In[16]:


y_train = np_utils.to_categorical(y_train, 2)
y_valid = np_utils.to_categorical(y_valid, 2)
y_test = np_utils.to_categorical(y_test, 2)


# In[17]:


params = dvc.api.params_show()


# In[18]:


early = EarlyStopping(monitor="val_categorical_accuracy", patience=params["train"]["patience"])


# In[19]:


def smart_model():
    shape = (192, 192, 3)
    input_tensor = Input(shape=shape)
    base_model = MobileNet(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=shape,
        pooling='avg'
    )

    for layer in base_model.layers:
        layer.trainable = True

    op = Dense(params["model"]["dense"], activation='relu')(base_model.output)
    op = Dropout(params["model"]["dropout"])(op)

    output_tensor = Dense(2, activation='sigmoid')(op)

    model = Model(input_tensor, output_tensor)
    return model


# In[20]:


model = smart_model()


# In[21]:


model.compile(optimizer=SGD(),
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy', 
                       Precision(),
                       Recall(),
                       AUC()])


# In[ ]:


history = model.fit(x_train,
                    y_train,
                    epochs=params["train"]["epochs"],
                    batch_size=params["train"]["batch"],
                    validation_data=[x_valid, y_valid],
                    callbacks=[early])


# In[ ]:


score = model.evaluate(x_test, y_test)
scores = {"Accuracy": score[1], "Precision": score[2], "Recall": score[3], "AUC": score[4]}


# In[ ]:


model.save("./models/model.h5")


# In[ ]:


with open(path_scores, 'w') as file:
           json.dump(scores, file)

