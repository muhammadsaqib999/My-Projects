#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Model

# In[26]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\Housing.csv")
a.head()


# In[27]:


a.columns


# In[28]:


a.shape


# In[29]:


a["mainroad"].unique(),a["mainroad"].value_counts()


# # Perform Label-Encoding

# In[30]:


a.head()


# In[31]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()


# In[32]:


a["mainroad"]=l.fit_transform(a["mainroad"])
a["guestroom"]=l.fit_transform(a["guestroom"])
a["basement"]=l.fit_transform(a["basement"])
a["hotwaterheating"]=l.fit_transform(a["hotwaterheating"])
a["airconditioning"]=l.fit_transform(a["airconditioning"])
a["prefarea"]=l.fit_transform(a["prefarea"])
a["furnishingstatus"]=l.fit_transform(a["furnishingstatus"])


# In[33]:


a.head()


# In[34]:


x=a.drop(columns=["price"])
y=a["price"]


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)


# In[36]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[37]:


lr.fit(x_train,y_train)


# In[38]:


from sklearn.model_selection import cross_val_score,KFold
cr=cross_val_score(lr,x,y,cv=KFold(n_splits=10))


# In[39]:


cr


# In[40]:


lr.score(x_test,y_test)*100


# In[41]:


lr.predict([[7420,4,2,3,1,0,0,0,1,2,1,0]])


# In[42]:


a.head()


# In[43]:


cor=a.corr()


# In[44]:


sns.heatmap(cor, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size': 8}, cmap="Blues")


# In[45]:


sns.pairplot(data=a,palette=['red'])
plt.show()


# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt

for col in ['area', 'bedrooms', 'bathrooms']:  # Add other variables as needed
    sns.scatterplot(x=a[col], y=a['price'])
    plt.title(f"Price vs {col}")
    plt.show()


# In[48]:


from sklearn.preprocessing import PolynomialFeatures
p=PolynomialFeatures(degree=2)
p.fit(x)
p.transform(x)


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)


# In[50]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[51]:


lr.fit(x_train,y_train)


# In[52]:


lr.score(x_test,y_test)*100


# In[55]:


lr.predict([[9960,3,2,2,1,0,1,0,0,2,1,1]])


# In[54]:


a.head()


# In[ ]:




