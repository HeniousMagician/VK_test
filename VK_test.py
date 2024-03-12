#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re 

import pandas as pd 
import matplotlib.pyplot as plt 
import datetime 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# In[2]:


fixed_df_train = pd.read_csv('train_df.csv', delimiter=',')


# In[3]:


fixed_df_train = fixed_df_train.drop('search_id', axis=1)
X = fixed_df_train.drop('target', axis=1)
y = fixed_df_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[4]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


# Определение параметров
activation = 'relu'
hidden_layers = [100, 50]
loss = 'binary_crossentropy'
optimizer = 'adam'

# Создание модели
model = MLPClassifier(hidden_layer_sizes=[100, 50], activation='tanh', solver='sgd', learning_rate_init=0.1)


# In[8]:


model.fit(X_train, y_train)


# In[9]:


# Оценка точности
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Точность:", accuracy)


# In[10]:


fixed_df_test = pd.read_csv('test_df.csv', delimiter=',')


# In[11]:


# Определите метод ранжирования
fixed_df_test = fixed_df_test.sort_values(by='target', ascending=False)

# Инициализация
p = 3
DCG = []

# Расчет DCG
for i in range(fixed_df_test.shape[0]):
    relevance_position = i + 1
    gain = fixed_df_test['target'].iloc[i] * (1 / (p ** relevance_position))
    DCG.append(gain)

DCG = sum(DCG)
ideal_DCG = fixed_df_test['target'].sum() * (1 - (1 / (p ** (fixed_df_test.shape[0] + 1))))
NDCG = DCG / ideal_DCG

print("NDCG:", NDCG)


# In[ ]:




