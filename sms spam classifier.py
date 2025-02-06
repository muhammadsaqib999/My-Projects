#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\js\class 1\venv\Scripts\spam.csv", encoding='latin1')
df.head()


# In[6]:


df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df.rename(columns={"v1":"target","v2":"text"},inplace=True)


# In[10]:


df.head()


# In[11]:


df.isnull().sum()


# In[12]:


df["target"].value_counts()


# In[13]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["target"]=lb.fit_transform(df["target"])


# In[14]:


df.head()


# In[15]:


df.duplicated().sum()


# In[16]:


df=df.drop_duplicates(keep="first")


# In[17]:


df.shape


# In[18]:


df["target"].value_counts()


# In[19]:


plt.pie(df["target"].value_counts(),labels=["Ham","Spam"],autopct="%.2f")
plt.show()


# In[20]:


pip install nltk


# In[23]:


import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


df["num_characters"]=df["text"].apply(len)


# In[26]:


df.head()


# In[27]:


df["num_words"]=df["text"].apply(lambda x:len(nltk.word_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df["num_sentence"]=df["text"].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[30]:


df.head()


# In[31]:


sns.histplot(df[df["target"]==0]["num_characters"])
sns.histplot(df[df["target"]==1]["num_characters"],color="red")


# In[32]:


sns.histplot(df[df["target"]==0]["num_words"])
sns.histplot(df[df["target"]==1]["num_words"],color="red")


# In[33]:


sns.pairplot(df,hue="target")
plt.show()


# # Data Processing

# In[34]:


# Lower case
# Tokenize
# Removing Special characters
# pemoving stop words and punctuation
# stemming


# In[61]:


def transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
# remove special character    
    y=[]
    for i in text:
         if i.isalnum(): #isalnum mean alphanumeric or numeric
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[62]:


transform("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today")


# In[63]:


df["text"][10]


# In[64]:


import string
string.punctuation   # These are Special Characters


# In[65]:


import nltk
nltk.download('stopwords')  # These are stopwords


# In[66]:


from nltk.corpus import stopwords
stopwords.words('english')


# # Stemming

# In[78]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()
ps.stem("Dancing")


# In[77]:


df["transformed_text"]=df["text"].apply(transform)


# In[69]:


df.head()


# # Model Train/Test Apply Algorithm

# In[70]:


# First of all convert transformed_text columns to number(encoding) by the help of vectorization.
#Vectorizer as same as Encoding .Its function is to convert categorical data to numerical form


# In[71]:


#There are different form for data vectorizing (1:CountVectorizer,TfidfVectorizer etc)


# # CountVectorizer and TfidfVectorizer

# In[79]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#cv=CountVectorizer()
td=TfidfVectorizer(max_features=3000) #without "max_features=3000" it also run but "max_features=3000" improve accuracy.we can change value.
#x = cv.fit_transform(df["transformed_text"]).toarray()
x = td.fit_transform(df["transformed_text"]).toarray()


# In[80]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
x=scale.fit_transform(x)


# In[81]:


y=df["target"].values


# In[82]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# In[83]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[84]:


gnb.fit(x_train,y_train)
y_predict1=gnb.predict(x_test)
print(accuracy_score(y_test,y_predict1)*100)
print(confusion_matrix(y_test,y_predict1))
print(precision_score(y_test,y_predict1)*100)


# In[85]:


mnb.fit(x_train,y_train)
y_predict1=mnb.predict(x_test)
print(accuracy_score(y_test,y_predict1)*100)
print(confusion_matrix(y_test,y_predict1))
print(precision_score(y_test,y_predict1)*100)


# In[86]:


bnb.fit(x_train,y_train)
y_predict1=bnb.predict(x_test)
print(accuracy_score(y_test,y_predict1)*100)
print(confusion_matrix(y_test,y_predict1))
print(precision_score(y_test,y_predict1)*100)


# # Improving Model

# In[87]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[88]:


svc=SVC(kernel="sigmoid",gamma=1.0)
kn=KNeighborsClassifier()
mnb=MultinomialNB()
dt=DecisionTreeClassifier(max_depth=5)
lr=LogisticRegression(solver='liblinear',penalty='l2')
rf=RandomForestClassifier(n_estimators=50,random_state=2)
adc=AdaBoostClassifier(n_estimators=50,random_state=2)
bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
gbc=GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[89]:


clfs={
    "SVC":svc,
    "K-NN":kn,
    "Multinomial-NB":mnb,
    "D-Tree-classifier":dt,
    "Logistic-regression":lr,
    "Random_forestClassifier":rf,
    "AdaBoostClassifier":adc,
    "BaggingClassifier":bc,
    "ExtraTreesClassifier":etc,
    "GradientBoostingClassifier":gbc
    
}


# In[90]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=50)


# In[91]:


def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)*100
    precision=precision_score(y_test,y_pred)*100
    
    return accuracy,precision


# In[95]:


train_classifier(svc,x_train,y_train,x_test,y_test)


# In[96]:


from sklearn.metrics import accuracy_score, precision_score

# Avoid using names that clash with function names
accuracies = []
precisions = []

def train_classifier(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100  # sklearn.metrics function
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    return accuracy, precision

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, x_train, y_train, x_test, y_test)
    accuracies.append(current_accuracy)
    precisions.append(current_precision)
    print("For", name, "Accuracy:", current_accuracy, "Precision:", current_precision)


# In[97]:


performence_df=pd.DataFrame({"Algorithms":clfs.keys(),"Accuracy":current_accuracy,"Precision":current_precision})


# In[98]:


performence_df = performence_df.sort_values(by="Accuracy", ascending=False)


# In[102]:


performence_df


# In[106]:


import pickle
pickle.dump(td,open("vect.pkl","wb"))
pickle.dump(bnb,open("mod.pkl","wb"))


# In[ ]:





# In[ ]:





# In[107]:


get_ipython().system('pip install scikit-learn')


# In[ ]:




