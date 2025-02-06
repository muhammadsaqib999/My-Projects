#!/usr/bin/env python
# coding: utf-8

# # Loan Status Prediction Machine Learning Project

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\loan.csv")
a.head(10)


# In[3]:


sns.countplot(x="Dependents",hue="Loan_Status",data=a)
plt.show()


# In[4]:


a.head()


# In[5]:


from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

a["Gender"] = lb.fit_transform(a["Gender"])
a["Self_Employed"] = lb.fit_transform(a["Self_Employed"])
a["Married"] = lb.fit_transform(a["Married"])
a["Education"] = lb.fit_transform(a["Education"])
a["Property_Area"] = lb.fit_transform(a["Property_Area"])
a["Loan_Status"] = lb.fit_transform(a["Loan_Status"])
a=a.replace(to_replace="3+",value=3)


# In[6]:


a=a.replace(to_replace="3+",value=3)


# In[7]:


a.head(10)


# In[8]:


a=a.fillna(120)


# In[9]:


a.head()


# In[10]:


a.drop(columns=["Loan_ID"],inplace=True)


# In[11]:


a.head()


# In[12]:


x=a.iloc[:,:-1]
y=a["Loan_Status"]


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# # Apply SVM Algorithm

# In[14]:


from sklearn.svm import SVC
sv=SVC(kernel="poly")
sv.fit(x_train,y_train)


# In[15]:


sv.score(x_test,y_test)*100


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=30)
kn.fit(x_train,y_train)


# In[17]:


kn.score(x_test,y_test)*100


# In[18]:


sv.predict([[7,1,1,3,0,0,3036,2504.0,158.0,360.0,0.0]])


# In[19]:


kn.predict([[7,1,1,3,0,0,3036,2504.0,158.0,360.0,0.0]])


# In[20]:


a.head(10)


# In[ ]:







# In[ ]:





# In[ ]:





# In[21]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)


# In[22]:


dt.predict([[1,0,0,0,0,5849,0.0,120.0,360.0,1.0,2]])


# In[56]:


kn.score(x_test,y_test)*100


# In[64]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,dt.predict(x_test))
c


# In[65]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,sv.predict(x_test))
c


# In[66]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,kn.predict(x_test))
c


# # Decision Tree Algorithm is best to use Because its Accuracy is high according to Confision_Matrix
