#!/usr/bin/env python
# coding: utf-8

# # Breast_Cancer Prediction Deep Learning Project
# 

# In[97]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[98]:


a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\breast-cancer.csv")


# In[99]:


a.head()


# In[100]:


a.drop(columns=["id"],inplace=True)


# In[101]:


from sklearn.preprocessing import LabelEncoder
new=LabelEncoder()


# In[102]:


a['diagnosis'] =new.fit_transform(a['diagnosis'])


# In[103]:


a.tail()


# In[104]:


a.head()


# In[106]:


a.shape


# In[107]:


a.describe()


# In[110]:


a["diagnosis"].value_counts()


# In[111]:


a.isnull().sum()


# In[112]:


a.duplicated().sum()


# In[113]:


a.groupby("diagnosis").mean()


# In[116]:


x=a.drop(columns=["diagnosis"],axis=1)
y=a["diagnosis"]


# In[117]:


x.shape,y.shape


# # Train/Test Split

# In[139]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[140]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[190]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_std=scale.fit_transform(x_train)
x_test_std=scale.transform(x_test)


# In[191]:


print(x_train_std)


# # Neural Network Architecture

# In[192]:


import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


# In[193]:


model=keras.Sequential([
                        keras.layers.Flatten(input_shape=(30,)),
                        keras.layers.Dense(20,activation="relu"),
                        keras.layers.Dense(2,activation="sigmoid")
    
]) 


# In[194]:


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[195]:


history=ann.fit(x_train_std,y_train,validation_split=0.1,epochs=10)


# # Visualization

# In[196]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")

plt.legend(['training_data','validation_data'],loc='center right')


# In[197]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("model loss")
plt.ylabel("accuracy")
plt.xlabel("epoch")

plt.legend(['training_data', 'validation_data'], loc='center right')


# # Accuracy of the Model

# In[200]:


loss,accuracy=model.evaluate(x_test_std,y_test)
print(accuracy)


# In[205]:


print(x_test_std.shape)
print(x_test_std[0])


# In[206]:


y_pred=model.predict(x_test_std)


# In[211]:


print(x_test_std.shape)
print(y_pred)


# In[212]:


y_pred_label=[np.argmax(i) for i in y_pred]
print(y_pred_label)


# # Prediction on Data

# In[220]:


input_data=(17.2,24.52,114.2,929.4,0.1071,0.183,0.1692,0.07944,0.1927,0.06487,0.5907,1.041,3.705,69.47,0.00582,0.05616,0.04252,0.01127,0.01527,0.006299,23.32,33.82,151.6,1681,0.1585,0.7394,0.6566,0.1899,0.3313,0.1339)

input_data_to_numpy_array=np.asarray(input_data)

input_data_reshape=input_data_to_numpy_array.reshape(1,-1)

input_data_std_after_reshape=scale.transform(input_data_reshape)

prediction=model.predict(input_data_std_after_reshape)
print(prediction)

prediction_label=[np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0]==0):
    print("Non_Malignant")
else:
    print("Malignant")


# In[ ]:




