# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## DESIGN STEPS

### STEP 1:
Load the csv file and then use the preprocessing steps to clean the data

### STEP 2:
Split the data to training and testing

### STEP 3:
Train the data and then predict using Tensorflow.

## PROGRAM
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import sklearn
df = pd.read_csv('customers.csv')
df.isnull().sum()
df_cleaned=df.drop(["ID","Var_1"],axis=1)
df_cleaned.head()
df_cleaned= df_cleaned.dropna(axis=0)
df_cleaned.isnull().sum()
df_col=list(df_cleaned.columns)
data_col_obj=list()
for c in df_col:
  if df_cleaned[c].dtype=='O':
      data_col_obj.append(c)
      
data_col_obj.remove("Segmentation")

data_col_obj
df_cleaned[data_col_obj]=OrdinalEncoder().fit_transform(df_cleaned[data_col_obj])
df_cleaned[["Age"]]=MinMaxScaler().fit_transform(df_cleaned[["Age"]])
df_cleaned.head()
df_cleaned['Segmentation'] = LabelEncoder().fit_transform(df_cleaned['Segmentation'])
y=df_cleaned[['Segmentation']].values
y = OneHotEncoder().fit_transform(y).toarray()
X=df_cleaned.iloc[:,:-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=50)
model = Sequential([Dense(48,input_shape=(8,),activation='relu'),
                      Dense(32,activation='relu'),
                       Dense(8,activation='relu'),
                       Dense(4,activation='softmax'),
])
model.compile(optimizer='Adagrad',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2,restore_best_weights=True)
model.fit(x=X_train,y=y_train,
             epochs=5000,batch_size=256,
             validation_data=(X_test,y_test),
             callbacks=[early_stop]
             )
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
y_preds=tf.argmax(model.predict(X_test),axis=1)
print(classification_report(tf.argmax(y_test,axis=1),y_preds))
print(confusion_matrix(tf.argmax(y_test,axis=1),y_preds))
tf.argmax(model.predict([[0., 0., 0., 6., 0.,3.,5.,6.]]),axis=1)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/75235488/189712954-ea5e7d8a-a774-458d-91de-5eb299f53f1e.png)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235488/189713155-b7103658-6bd0-46c6-93c2-6f1b063857ad.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235488/189713249-3e258de2-a21e-46a2-b060-1877fd0fae48.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235488/189713310-ebeb788f-1b20-49d2-950a-f90058a4e127.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235488/189713377-7bf356d9-150b-483d-837e-166197d0a654.png)

## RESULT
Thus a Neural Network Classification Model is created and executed successfully
