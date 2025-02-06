#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Deep Learning Project

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\customer_churn_prediction_dataset.csv")
a.tail()


# In[4]:


a.shape


# In[6]:


a.isnull().sum()


# In[7]:


a.duplicated().sum()


# In[9]:


a.describe()


# In[19]:


a["gender"].value_counts()


# In[20]:


a["SeniorCitizen"].value_counts()


# In[21]:


a["PhoneService"].value_counts()


# In[23]:


a.drop(columns="customerID",inplace=True)


# In[24]:


a.head()


# # Label Encoding

# In[25]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
a["gender"]=lb.fit_transform(a["gender"])
a["Partner"]=lb.fit_transform(a["Partner"])
a["Dependents"]=lb.fit_transform(a["Dependents"])
a["PhoneService"]=lb.fit_transform(a["PhoneService"])
a["MultipleLines"]=lb.fit_transform(a["MultipleLines"])
a["InternetService"]=lb.fit_transform(a["InternetService"])
a["OnlineSecurity"]=lb.fit_transform(a["OnlineSecurity"])
a["OnlineBackup"]=lb.fit_transform(a["OnlineBackup"])
a["DeviceProtection"]=lb.fit_transform(a["DeviceProtection"])
a["TechSupport"]=lb.fit_transform(a["TechSupport"])
a["StreamingTV"]=lb.fit_transform(a["StreamingTV"])
a["StreamingMovies"]=lb.fit_transform(a["StreamingMovies"])
a["Contract"]=lb.fit_transform(a["Contract"])
a["PaperlessBilling"]=lb.fit_transform(a["PaperlessBilling"])
a["PaymentMethod"]=lb.fit_transform(a["PaymentMethod"])
a["Churn"]=lb.fit_transform(a["Churn"])


# In[26]:


a.head()


# In[31]:


x=a.drop(columns="Churn")
y=a["Churn"]


# # Perform Scaling

# In[77]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=pd.DataFrame(ss.fit_transform(x),columns=x.columns)
x_train_scaled=ss.transform(x_train) 


# In[171]:


x_train_scaled


# In[64]:


x.head()


# In[65]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# # Making Neural Network

# In[129]:


import tensorflow
from keras.layers import Dense,BatchNormalization,Dropout
from keras.models import Sequential


# In[130]:


ann=Sequential()


# In[131]:


ann.add(Dense(8,activation="relu",input_dim=19))
ann.add(Dropout(0.4))
ann.add(BatchNormalization())

ann.add(Dense(6,activation="relu"))
ann.add(Dropout(0.3))
ann.add(BatchNormalization())

ann.add(Dense(4,activation="relu"))
ann.add(Dropout(0.2))
ann.add(BatchNormalization())

ann.add(Dense(2,activation="relu"))
ann.add(Dropout(0.1))
ann.add(BatchNormalization())

ann.add(Dense(1,activation="sigmoid"))


# In[172]:


ann.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[173]:


ann.summary()


# In[174]:


ff=ann.fit(x_train_scaled, y_train,epochs=70)


# # Test Accuracy

# In[184]:


test_prd=ann.predict(x_test)


# In[176]:


test_prd_data=[]
for i in test_prd:
    if i[0]>0.5:
        test_prd_data.append(1)
    else:
        test_prd_data.append(0)


# In[177]:


ann.predict([[0.993355,-1.090771,0.967204,0.948016,-1.460436,-1.040833,1.230129,1.228038,-1.175205,1.193145,-1.280034,-1.263353,-1.091744,-0.004156,-1.142059,0.910642,-0.439460,-1.391968,-1.222146]])


# In[178]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,test_prd_data)*100


# In[179]:


ann.predict([[-1.006689,-1.090771,-1.033908,-1.054835,0.294934,0.960769,0.020166,-1.252847,1.239600,1.193145,-1.280034,-1.263353,0.134935,-0.004156,1.296502,-1.098127,0.457398,-1.012469,-0.454340]])


# # Train Accuracy

# In[180]:


train_prd=ann.predict(x_train)


# In[181]:


train_prd_data=[]
for i in train_prd:
    if i[0]>0.5:
        train_prd_data.append(1)
    else:
        train_prd_data.append(0)


# In[182]:


accuracy_score(y_train,train_prd_data)*100


# In[165]:


plt.plot(ff.history['accuracy'],color="g")


# In[158]:





# In[ ]:




