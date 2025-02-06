#!/usr/bin/env python
# coding: utf-8

# # Parkinsons Disease Prediction (Machine Learning Model)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a = pd.read_csv('parkinsons.data')

a.head()


# In[47]:


sns.countplot(x="MDVP:Fo(Hz)",hue="status",data=a)
plt.show()


# In[2]:


a.drop(columns="name",inplace =True)


# In[10]:


x=a.iloc[:,:-1]
y=a["status"]


# In[11]:


a.shape


# In[12]:


a["status"].value_counts()


# # Apply imbalance Algorithm

# In[13]:


from imblearn.under_sampling import RandomUnderSampler
r=RandomUnderSampler()


# In[14]:


r_x,r_y=r.fit_resample(x,y)


# In[15]:


r_y.value_counts()


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(r_x,r_y,test_size=0.1,random_state=50)


# # Apply SVC Algorithm

# In[21]:


from sklearn.svm import SVC
sv=SVC(kernel="poly")
sv.fit(r_x,r_y)


# In[22]:


sv.score(x_test,y_test)*100


# # Apply Logistic-Regression Algorithm

# In[26]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(r_x,r_y)


# In[27]:


lr.score(x_test,y_test)*100


# # Apply K-NN Algorithm

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=20)
kn.fit(r_x,r_y)


# In[29]:


kn.score(r_x,r_y)*100


# # Apply Decision-Tree Algorithm

# In[30]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=25)
dt.fit(r_x,r_y)


# In[31]:


dt.score(x_test,y_test)*100


# # Prediction

# In[39]:


dt.predict([[243.43900,250.91200,232.43500,0.00210,0.000009,0.00109,0.00137,0.00327,0.01419,0.12600,0.00777,0.00898,0.01033,0.02330,0.00454,25.36800,0.438296,0.635285,-7.057869,0.091608,2.330716,0.091470]])


# In[40]:


dt.predict([[243.43900,250.91200,232.43500,0.00210,0.000009,0.00109,0.00137,0.00327,0.01419,0.12600,0.00777,0.00898,0.01033,0.02330,0.00454,25.36800,0.438296,0.635285,-7.057869,0.091608,2.330716,0.091470]])


# In[42]:


dt.predict([[243.43900,250.91200,232.43500,0.00210,0.000009,0.00109,0.00137,0.00327,0.01419,0.12600,0.00777,0.00898,0.01033,0.02330,0.00454,25.36800,0.438296,0.635285,-7.057869,0.091608,2.330716,0.091470]])


# In[41]:


dt.predict([[243.43900,250.91200,232.43500,0.00210,0.000009,0.00109,0.00137,0.00327,0.01419,0.12600,0.00777,0.00898,0.01033,0.02330,0.00454,25.36800,0.438296,0.635285,-7.057869,0.091608,2.330716,0.091470]])


# # Confusion Matrix

# In[43]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,sv.predict(x_test))
c


# In[44]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,lr.predict(x_test))
c


# In[45]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,kn.predict(x_test))
c


# In[46]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,dt.predict(x_test))
c


# In[ ]:




