#!/usr/bin/env python
# coding: utf-8

# # Diabeties Machine Learning Project(Sugar Patient Determination)(Classification)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\diabetes.csv")
a.head(20)


# In[2]:


x=a[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y=a["Outcome"]


# In[3]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
sc.transform(x)


# In[4]:


z=pd.DataFrame(sc.transform(x),columns=x.columns)


# In[5]:


z.head(20)


# In[6]:


f=a.head(50)


# In[7]:


sns.pairplot(data=f,hue="Outcome")
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# In[9]:


from sklearn.svm import SVC
sv=SVC(kernel="poly")
sv.fit(x_train,y_train)


# In[10]:


sv.score(x_train,y_train)*100,sv.score(x_test,y_test)*100


# In[11]:


sv.predict([[3,78,50,32,88,31,0.248,26]])


# In[12]:


a["Outcome"].value_counts()


# In[13]:


pip install imbalanced-learn


# In[14]:


pip install --upgrade imbalanced-learn


# In[16]:


from imblearn.under_sampling import RandomUnderSampler
print("imbalanced-learn imported successfully!")


# In[17]:


from imblearn.under_sampling import RandomUnderSampler
ru=RandomUnderSampler(random_state=42)
ru_x,ru_y=ru.fit_resample(x,y)


# In[18]:


ru_y.value_counts()


# In[19]:


x=a[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y=a["Outcome"]


# In[20]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
sc.transform(x)


# In[21]:


z=pd.DataFrame(sc.transform(x),columns=x.columns)


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# # SVC Algorithm

# In[27]:


from sklearn.svm import SVC
sv=SVC(kernel="linear")
sv.fit(x_train,y_train)


# In[28]:


sv.score(x_train,y_train)*100,sv.score(x_test,y_test)*100


# In[29]:


sv.predict([[0.936914,-0.434859,0.253036,-1.288212,-0.692891,-0.303664,-0.658012,-0.190672]])


# # K-NN Algorithm

# In[30]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=10)
kn.fit(x_train,y_train)


# In[31]:


kn.score(x_train,y_train)*100,kn.score(x_test,y_test)*100


# In[32]:


kn.predict([[0.936914,-0.434859,0.253036,-1.288212,-0.692891,-0.303664,-0.658012,-0.190672]])


# # Confusion Matrix for K-NN Algorithm

# In[36]:


from sklearn.metrics import confusion_matrix
g=confusion_matrix(y_test,kn.predict(x_test))


# In[37]:


g


# # Confusion Matrix for SVC Algorithm

# In[38]:


from sklearn.metrics import confusion_matrix
h=confusion_matrix(y_test,sv.predict(x_test))


# In[39]:


h


# In[40]:


input_data=(0.936914,-0.434859,0.253036,-1.288212,-0.692891,-0.303664,-0.658012,-0.190672)
input_data_as_numpyarray=np.asarray(input_data)
reshape=input_data_as_numpyarray.reshape(1,-1)
prediction=kn.predict(reshape)
print(prediction)
if (prediction[0]==0):
    print("Non Diabetic")
else:
    print("Diabetic")



# In[41]:




