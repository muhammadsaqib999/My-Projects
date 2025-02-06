#!/usr/bin/env python
# coding: utf-8

# # Insurance Medical Credit

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\insurance.csv")
a.head()


# In[61]:


a.isnull().sum()


# In[4]:


a.describe()


# In[60]:


sns.countplot(x="bmi",data=a)
plt.show()


# In[8]:


a.head()


# In[18]:


sns.pairplot(data=a)
plt.show()


# In[9]:


a.shape


# In[10]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[11]:


a["sex"]=lb.fit_transform(a["sex"])
a["smoker"]=lb.fit_transform(a["smoker"])
a["region"]=lb.fit_transform(a["region"])


# In[12]:


a.head()


# In[50]:


x=a.drop(columns=["charges"])
y=a["charges"]


# In[66]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)


# # Apply Linear Regression

# In[67]:


from sklearn.linear_model import LinearRegression
l=LinearRegression()
l.fit(x_train,y_train)


# In[68]:


l.score(x_train,y_train)*100,l.score(x_test,y_test)*100


# In[69]:


l.predict([[18,1,33.770,1,0,2]])


# In[70]:


a.head()


# # Apply Lasso Regression Algorithm

# In[56]:


from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(x_train, y_train)


# In[57]:


ls.score(x_train,y_train)*100,ls.score(x_test,y_test)*100


# In[58]:


l.predict([[18,1,33.770,1,0,2]])


# In[ ]:




