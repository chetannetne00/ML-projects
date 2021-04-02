#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


email = pd.read_csv('spam.csv',encoding='latin-1')
email.head()


# In[3]:


email.shape


# In[4]:


email.columns


# In[5]:


email.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[6]:


email.shape


# In[7]:


email.rename(columns={"v1":"label","v2":"mails"},inplace=True)


# In[8]:


email['label']=email['label'].map({'spam':1,'ham':0})


# In[9]:


train_data = email.iloc[:4179,:]
test_data = email.iloc[4179:,:]


# In[10]:


train_data.shape


# In[11]:


test_data.shape


# In[12]:


spam_words = ' '.join


# In[13]:


get_ipython().system('pip install nltk')
import nltk
#from wordcloud import WordCloud
#spam_wc = WordCloud(width=512,height=512).generate(sapm_words)


# In[14]:


#spam_words = ' '.join(list(email[email['label']==1]['mails']))


# In[15]:


import re
nltk.download('stopwords')


# In[16]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[17]:


data = [' i am Aniket Jain a civil 3 @ # $5623 engineeing Student','this is gonna Be So intersting and Amazing']
#here i have define a corpus by removing stopwords & stemming them
qwerty=[]
corpus=[]
for i in range(len(email['mails'])):
    qwerty= re.sub('[^a-zA-Z]',' ',email['mails'][i])
    qwerty.lower()
    qwerty= qwerty.split()
    qwerty=[ps.stem(word) for word in qwerty if not word in stopwords.words('english')]
    qwerty=' '.join(qwerty)
    corpus.append(qwerty)
    
     


# In[21]:


# learn to apply lemmitization
data = [' i am Aniket Jain a civil 3 @ # $5623 engineeing Student','this went gonna Be So intersting and Amazing']
b=ps.stem(data[1])
b


# In[22]:


email['corpus']=corpus
email.head()


# In[ ]:


get_ipython().system('conda install -c conda-forge wordcloud==1.4.1 --yes')


# In[20]:


data = email
data.head()


# In[21]:


train_data=data.iloc[:4179,:]
test_data=data.iloc[4179:,:]


# In[82]:


output= train_data.to_csv('emails_data.csv',index=True)
output


# In[78]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
x = CountVectorizer()
train_features = x.fit_transform(train_data['corpus']).toarray()
test_features = x.transform(test_data['corpus']).toarray()
train_features


# In[68]:


train_features.to_Pd.DataFrame


# In[24]:


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(train_features,train_data['label'])
pred= clf.predict(test_features)
pred


# In[42]:


Q=clf.predict_proba(train_features)
Q


# In[52]:


label = train_data['label']
label = np.array(label)
label
P=[[1-label[i],label[i]] for i in range(len(label))]


# In[53]:


import math
#Cross_entropy
def cross_entropy(P,Q):
    H = -1 * sum([P[i]* math.log(Q[i]+ 1e-15,2) for i in range(len(P))])
    return H


# In[64]:


cr_entr = [cross_entropy(P[i],Q[i]) for i in range(len(P))]
np.mean(cr_entr)


# In[61]:





# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix
acc = accuracy_score(test_data['label'],pred)
matrix=confusion_matrix(test_data['label'],pred)
print(acc,matrix)


# In[26]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
x = TfidfVectorizer()
train_features = x.fit_transform(train_data['corpus']).toarray()
test_features = x.transform(test_data['corpus']).toarray()

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(train_features,train_data['label'])
pred= clf.predict(test_features)

from sklearn.metrics import accuracy_score,confusion_matrix
acc = accuracy_score(test_data['label'],pred)
matrix=confusion_matrix(test_data['label'],pred)
print(acc,matrix)


# In[27]:


np.shape(train_features)


# In[28]:


np.shape(test_features)


# In[ ]:




