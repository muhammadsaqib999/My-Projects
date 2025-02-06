#!/usr/bin/env python
# coding: utf-8

# # Submarine Machine Learning Project (Rock & Mine Determination)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\sonar.csv",header=None)   
a


# In[2]:


a.shape


# In[3]:


a[60].value_counts()


# In[4]:


a.isnull().sum()


# In[5]:


a.describe()


# In[6]:


x=a.drop(columns=60,axis=1)
y=a[60]


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=20)


# In[8]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[9]:


lr.fit(x_train,y_train)


# In[10]:


lr.score(x_train,y_train)*100,lr.score(x_test,y_test)*100


# In[12]:


lr.predict([[0.0134,0.0172,0.0178,0.0363,0.0444,0.0744,0.0800,0.0456,0.0368,0.1250,0.2405,0.2325,0.2523,0.1472,0.0669,0.1100,0.2353,0.3282,0.4416,0.5167,0.6508,0.7793,0.7978,0.7786,0.8587,0.9321,0.9454,0.8645,0.7220,0.4850,0.1357,0.2951,0.4715,0.6036,0.8083,0.9870,0.8800,0.6411,0.4276,0.2702,0.2642,0.3342,0.4335,0.4542,0.3960,0.2525,0.1084,0.0372,0.0286,0.0099,0.0046,0.0094,0.0048,0.0047,0.0016,0.0008,0.0042,0.0024,0.0027,0.0041]])


# In[46]:


sns.scatterplot(data=a)
plt.show()


# In[ ]:




