#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction ML model (Regression)

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\car.csv")
a.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[9]:


a["Car_Name"]=lb.fit_transform(a["Car_Name"])
a["Fuel_Type"]=lb.fit_transform(a["Fuel_Type"])
a["Seller_Type"]=lb.fit_transform(a["Seller_Type"])
a["Transmission"]=lb.fit_transform(a["Transmission"])


# In[11]:


a.tail()


# In[13]:


sns.pairplot(data=a)
plt.show()


# In[14]:


a.shape


# In[15]:


x=a.drop(columns=["Selling_Price"])
y=a["Selling_Price"]


# In[21]:


x.head()


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# In[19]:


from sklearn.linear_model import LinearRegression
l=LinearRegression()
l.fit(x_train,y_train)


# In[20]:


l.score(x_train,y_train)*100,l.score(x_test,y_test)*100


# In[37]:


l.predict([[68,2017,9.85,6900,2,0,1,0]])


# # Lasso Regression Algorithm

# In[27]:


from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(x_train,y_train)


# In[32]:


lasso.score(x_train,y_train)*100,lasso.score(x_test,y_test)*100


# In[36]:


lasso.predict([[68,2017,9.85,6900,2,0,1,0]])


# In[ ]:





# In[30]:


a.head()


# # Linear-Regression is best to use 
