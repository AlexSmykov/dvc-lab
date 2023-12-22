#!/usr/bin/env python
# coding: utf-8

# ## Модуль проверки и выгрузки данных в csv-формат

# In[100]:


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


import warnings
warnings.filterwarnings("ignore")


# In[101]:


path_train = "./data/training/"
path_valid = "./data/validation/"
path_test = "./data/evaluation/"


# In[102]:


def get_data(path):
    food = []
    non_food = []
    images = os.listdir(path + "/food")
    for img in images:
        food.append(path + "food/" + img)
    non_images = os.listdir(path + "/non_food")
    for img in non_images:
        non_food.append(path + "non_food/" + img)
    output = [i for i in food] + [i for i in non_food]
    marks = [0 for _ in range(len(output)//2)] + [1 for _ in range(len(output)//2)]
    return output, marks


# In[103]:


x_train, y_train = get_data(path_train)
x_valid, y_valid = get_data(path_valid)
x_test, y_test = get_data(path_test)


# In[104]:


print("Размер тренировочной выборки: ", len(x_train))
print("Размер валидационной выборки: ", len(x_valid))
print("Размер тестовой выборки: ", len(x_test))
print("Сколько съедобного/несъедобного в тренировочной: ", len([i for i in y_train if i == 1]), "/", 
     len([i for i in y_train if i == 0]))
print("Сколько съедобного/несъедобного в валидационной: ", len([i for i in y_valid if i == 1]), "/", 
     len([i for i in y_valid if i == 0]))
print("Сколько съедобного/несъедобного в тестовой: ", len([i for i in y_test if i == 1]), "/", 
     len([i for i in y_test if i == 0]))


# In[105]:


train = pd.DataFrame(columns=["images_src", "mark"])
valid = pd.DataFrame(columns=["images_src", "mark"])
test = pd.DataFrame(columns=["images_src", "mark"])


# In[106]:


train["images_src"] = x_train
train["mark"] = y_train
valid["images_src"] = x_valid
valid["mark"] = y_valid
test["images_src"] = x_test
test["mark"] = y_test


# In[108]:


train.to_csv("./csv_storage/train.csv", index=False)
valid.to_csv("./csv_storage/valid.csv", index=False)
test.to_csv("./csv_storage/test.csv", index=False)

