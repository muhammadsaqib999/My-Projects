#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction ( Made by Saqib-Arif "Ai")

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\heart_disease.csv")
a


# In[49]:


sns.catplot(x="thalach", data=a, color="red", height=3, aspect=1)
plt.show()


# In[11]:


x=a.drop(columns="target",axis=1)
y=a["target"]


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# # Apply SVM Algorithm

# In[16]:


from sklearn.svm import SVC
sv=SVC(kernel="poly")
sv.fit(x_train,y_train)


# In[17]:


sv.score(x_test,y_test)*100


# # Apply K-NN Algorithm

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=20)
kn.fit(x_train,y_train)


# In[30]:


kn.score(x_test,y_test)*100


# # Apply Decision-Tree Algorithm

# In[41]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=25)
dt.fit(x_train, y_train)


# In[42]:


dt.score(x_test,y_test)*100


# In[26]:


sv.predict([[67,1,0,160,286,0,0,108,1,1.5,1,3,2]])


# In[27]:


sv.predict([[67,1,0,160,286,0,0,108,1,1.5,1,3,2]])


# In[28]:


sv.predict([[67,1,0,160,286,0,0,108,1,1.5,1,3,2]])


# # Confusion Matrix

# In[43]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,sv.predict(x_test))
c


# In[44]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,kn.predict(x_test))
c


# In[45]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,dt.predict(x_test))
c


# # Decision Tree Algorithm is Best to use according to Confusion Matrix

# In[ ]:




