#!/usr/bin/env python
# coding: utf-8

# # Wine-Quality Prediction Machine Learning Model

# In[65]:


import pandas as pd
a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\wine_quality.csv")
a.head()


# # Convert Quality Column to Boolean Type(0 or 1)

# In[66]:


import pandas as pd

# Assuming 'a' is your DataFrame
threshold = 6  # Define your threshold
a['quality_binary'] = (a['quality'] >= threshold).astype(int)

# Check the transformed column
print(a['quality_binary'].value_counts())


# In[67]:


a.head()


# In[80]:


x=a.iloc[:,:-1]
y=a["quality_binary"]


# In[81]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# In[98]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=10)
kn.fit(x_train,y_train)


# In[99]:


kn.score(x_test,y_test)*100


# In[100]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=50)
dt.fit(x_train, y_train)


# In[101]:


kn.score(x_test,y_test)*100


# In[102]:


dt.predict([[5.7,1.13,0.09,1.5,0.172,7.0,19.0,0.9940000000000001,3.5,0.48,9.8,4]])


# In[103]:


kn.predict([[5.7,1.13,0.09,1.5,0.172,7.0,19.0,0.9940000000000001,3.5,0.48,9.8,4]])


# In[104]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,dt.predict(x_test))
c


# In[105]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,kn.predict(x_test))
c


# # Decision Tree Algorithm is best to use According to Confision Matrix
