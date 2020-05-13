#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import dependencies
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from numpy.random import seed
seed(1)


# In[2]:


#import cleaned data
get_ipython().run_line_magic('store', '-r target')
get_ipython().run_line_magic('store', '-r target_names')
get_ipython().run_line_magic('store', '-r data')


# In[3]:


#splitting testing and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)


# In[4]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScater model and fit it to the training data
X_scaler = StandardScaler().fit(X_train)


# In[5]:


# Transform the training and testing data using the X_scaler

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[6]:


from tensorflow.keras.utils import to_categorical


# In[7]:


#label and encode data set
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_test)
encoded_y_test = label_encoder.transform(y_test)
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)


# In[8]:


#one-hot encoding
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)


# In[9]:


#create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=9))
model.add(Dense(units=2, activation='softmax'))


# In[10]:


#summarize model
model.summary()


# In[11]:


#compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


#fit model
model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=100,
    shuffle=True,
    verbose=2
)


# In[13]:


#create 'deep' model
deep_model = Sequential()
deep_model.add(Dense(units=6, activation='relu', input_dim=9))
deep_model.add(Dense(units=6, activation='relu'))
deep_model.add(Dense(units=2, activation='softmax'))


# In[14]:


#deep model summary
deep_model.summary()


# In[15]:


#compile and fit deep model
deep_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

deep_model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=100,
    shuffle=True,
    verbose=2
)


# In[16]:


#test model
model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[17]:


#test deep model
model_loss, model_accuracy = deep_model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(f"Deep Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[ ]:





# In[ ]:




