# -*- coding: utf-8 -*-
"""Mask and Without Mask prediction Deep Learning(coolab)

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qXo_5c1RejgeLOdP1a4W2q2GbfXyJIiU

**Mask and Without Mask prediction Deep Learning(coolab)**
"""

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d omkargurav/face-mask-dataset

!unzip /content/face-mask-dataset.zip -d /content/

!ls

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
from sklearn.model_selection import train_test_split

with_mask_file =os.listdir("/content/data/with_mask")
print(with_mask_file[0:5])
print(with_mask_file[-5:])

without_mask_file =os.listdir("/content/data/without_mask")
print(without_mask_file[0:5])
print(without_mask_file[-5:])

print("Number of with mask images:",len(with_mask_file))
print("Number of without mask images:",len(without_mask_file))

"""image having mask will be--->> 1

image without mask will be-->> 0
"""

with_mask_label = np.ones(len(with_mask_file),dtype="int32")
print(with_mask_label)

without_mask_label = np.zeros(len(without_mask_file),dtype="int32")
print(without_mask_label)

print(with_mask_label[0:5])
print(with_mask_label[-5:])

print(without_mask_label[0:5])
print(without_mask_label[-5:])

# Concatenate the labels instead of adding them.
label = np.concatenate([with_mask_label, without_mask_label])

print(len(label))
print(label[0:5])
print(label[-5:])

print(len(with_mask_label))
print(len(without_mask_label))

"""**Displaying image**"""

img=mpimg.imread("/content/data/with_mask/with_mask_335.jpg")
imgplot=plt.imshow(img)
plt.show()

img=mpimg.imread("/content/data/without_mask/without_mask_2425.jpg")
imgplot=plt.imshow(img)
plt.show()

"""**Image processing**

Convert image to numpy Array,
Resize image
"""

with_mask_path = "/content/data/with_mask/"
data=[]
for img_file in with_mask_file:
  image = Image.open(with_mask_path + img_file)
  image = image.resize((128,128))
  image=image.convert("RGB")
  image = np.array(image)
  data.append(image)



without_mask_path = "/content/data/without_mask/"
for img_file in without_mask_file:
  image = Image.open(without_mask_path + img_file)
  image = image.resize((128,128))
  image=image.convert("RGB")
  image = np.array(image)
  data.append(image)

len(data)

type(data[0])

data[0]

data[0].shape

"""**Converting data and labels to numpy array**"""

x=np.array(data) #hamare pas data and label dono list form me te isi liye isko hm ne numpy.array me convert kiya
y=np.array(label)

type(x),type(y)

print(x.shape)
print(y.shape)

"""**Train Test**"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Scaling data
x_train_scaled = x_train/255
x_test_scaled = x_test/255

"""**Make Convolutional Layer (CNN)**"""

import tensorflow as tf
from tensorflow import keras

num_of_class = 2
model = keras.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64,activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_of_class,activation="sigmoid"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["acc"])

history=model.fit(x_train_scaled,y_train,validation_split=0.1,epochs=10)

loss,accuracy = model.evaluate(x_test_scaled,y_test)
print("Loss:",(loss)*100)
print("Accuracy:",(accuracy)*100)

h=history
plt.plot(h.history["acc"])
plt.plot(h.history["val_acc"])
plt.title("Accuraccy")
plt.legend(["train_acc","test_acc"])
plt.show()

plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.title("Loss")
plt.legend(["train_loss","test_loss"])
plt.show()

"""**Predictive System**"""

input_image_path = input("Enter the image path : ")
input_image=cv2.imread(input_image_path)
cv2_imshow(input_image)
input_image_resize = cv2.resize(input_image,(128,128))
input_image_scaled = input_image_resize/255
input_image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
print(input_prediction)
input_pred_label = np.argmax(input_prediction)
print(input_pred_label)
if input_pred_label == 1:
  print("The person is wearing mask")
else:
  print("The person is not wearing mask")

"""**Streamlit WEB APP**"""

model.save('/content/streamlit_app_making_mask_without_mask_dataset.keras')

!npm install localtunnel

!pip install streamlit

!streamlit run /content/app.py &>/content/logs.txt &

!npx localtunnel --port 8501

from os import path
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from google.colab.patches import cv2_imshow

st.header("Face Mask Detection")
Path=st.text_input("Enter image path")

input_image=cv2.imread(path)
cv2_imshow(input_image)
input_image_resize = cv2.resize(input_image,(128,128))
input_image_scaled = input_image_resize/255
input_image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
print(input_prediction)
input_pred_label = np.argmax(input_prediction)
print(input_pred_label)
if input_pred_label == 1:
  print("The person is wearing mask")
else:
  print("The person is not wearing mask")