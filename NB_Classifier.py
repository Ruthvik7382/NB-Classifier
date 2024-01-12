#!/usr/bin/env python
# coding: utf-8

# In[227]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import make_scorer, precision_score, recall_score


# In[116]:


import warnings
warnings.filterwarnings("ignore")


# In[117]:


df = pd.read_csv('malware.csv')


# In[118]:


df.head(5)


# In[119]:


df = df[['classification', 'os', 'usage_counter', 'prio', 'static_prio', 'normal_prio', 'vm_pgoff', 'vm_truncate_count', 'task_size', 'map_count', 'hiwater_rss', 'total_vm', 'shared_vm', 'exec_vm', 'reserved_vm', 'nr_ptes', 'nvcsw', 'nivcsw', 'signal_nvcsw']]


# In[120]:


df.shape


# In[121]:


df.head(5)


# In[122]:


df.isna().sum()


# In[123]:


y = df['classification'] 
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
df['classification'] = y_encoded


# In[124]:


df.shape


# In[125]:


df.head(5)


# In[126]:


df['classification'].unique()


# In[127]:


df.dtypes


# In[128]:


df_dummies = pd.get_dummies(df["os"])


# In[129]:


df = df.join(df_dummies)


# In[130]:


df.head(5)


# In[131]:


df = df.drop('os', axis=1)


# In[132]:


df = df.drop('CentOS', axis=1)


# In[133]:


df.head(5)


# In[94]:


df.shape


# In[135]:


df.isnull().sum(axis = 0)


# In[136]:


df.tail(5)


# In[137]:


df = df.drop(['usage_counter','normal_prio','vm_pgoff','task_size','hiwater_rss','nr_ptes','signal_nvcsw'],axis=1)


# In[138]:


df.head(5)


# In[139]:


y_encoded = df['classification']


# In[140]:


df = df.drop('classification', axis=1)


# In[141]:


df.head(5)


# In[102]:


cols = df.columns
df_num_std = pd.DataFrame(scaler.fit_transform(df), columns = cols)


# In[143]:


df_num_std.head(5)


# In[104]:


df_binary = df.copy(deep=True)
numCols = [0,1,2,3,4,5,6,7,8,9]
df_numerical = df_binary.iloc[:,numCols]
df_dummy = df_binary.drop(df_binary.columns[numCols], axis=1)


# In[105]:


df_numerical.head(5)


# In[106]:


df_dummy.head(5)


# In[110]:


df_numerical.dtypes


# In[145]:


df_numerical1 = df[['prio','static_prio','vm_truncate_count','map_count','total_vm','shared_vm','exec_vm','reserved_vm','nvcsw','nivcsw']]


# In[146]:


df_numerical1.dtypes


# In[147]:


group_names = ['L','M','H']
for col in df_numerical1.columns:
    df_numerical1[col] = pd.cut(df_numerical1[col], 3, labels=group_names)


# In[148]:


df_numerical1.head(5)


# In[150]:


df_numerical1['nivcsw'].unique()


# In[151]:


df_dummies1=pd.get_dummies(df_numerical1)


# In[152]:


df_dummies1.head(5)


# In[155]:


df_dummies1.shape


# In[153]:


cols1 = ['prio_L','static_prio_L','vm_truncate_count_L','map_count_L','total_vm_L','shared_vm_L','exec_vm_L','reserved_vm_L','nvcsw_L','nivcsw_L']


# In[156]:


df_dummies1 = df_dummies1.drop(cols1, axis=1)


# In[157]:


df_dummies1.shape


# In[162]:


df_encoded = pd.concat([df_dummies1,df_dummy],axis=1)


# In[177]:


df_encoded.head(5)


# In[164]:


y_encoded.head(5)


# In[172]:


y_encoded.shape


# In[169]:


df_encoded['prio_M'].unique()


# In[173]:


df_encoded.shape


# In[174]:


df.isnull().sum(axis = 0)


# In[237]:


# categorical NB


y = y_encoded
x = df_encoded
clf = CategoricalNB(alpha=1)

precision = make_scorer(precision_score, average='micro')
recall = make_scorer(recall_score, average='micro')
acc=cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()
prec=cross_val_score(clf, x, y, cv=10, scoring=precision).mean()
rec=cross_val_score(clf, x, y, cv=10, scoring=recall).mean()
print("N-fold Cross Validation: accuracy = ",acc,', precision = ', prec, ' , recall = ',rec)


# In[238]:





# In[240]:


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
x = df
y = y_encoded

precision = make_scorer(precision_score, average='micro')
recall = make_scorer(recall_score, average='micro')
acc=cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()
prec=cross_val_score(clf, x, y, cv=10, scoring=precision).mean()
rec=cross_val_score(clf, x, y, cv=10, scoring=recall).mean()
print("N-fold Cross Validation: accuracy = ",acc,', precision = ', prec, ' , recall = ',rec)


y = y_encoded
x = df_num_std
clf = GaussianNB()

precision = make_scorer(precision_score, average='micro')
recall = make_scorer(recall_score, average='micro')
acc=cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()
prec=cross_val_score(clf, x, y, cv=10, scoring=precision).mean()
rec=cross_val_score(clf, x, y, cv=10, scoring=recall).mean()
print("N-fold Cross Validation: accuracy = ",acc,', precision = ', prec, ' , recall = ',rec)


# In[235]:


from sklearn.naive_bayes import MultinomialNB 


y = y_encoded
x = df
clf = MultinomialNB()

precision = make_scorer(precision_score, average='micro')
recall = make_scorer(recall_score, average='micro')
acc=cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()
prec=cross_val_score(clf, x, y, cv=10, scoring=precision).mean()
rec=cross_val_score(clf, x, y, cv=10, scoring=recall).mean()
print("N-fold Cross Validation: accuracy = ",acc,', precision = ', prec, ' , recall = ',rec)


# In[242]:


from sklearn.naive_bayes import ComplementNB 

y = y_encoded
x = df
clf = ComplementNB(alpha=0)
precision = make_scorer(precision_score, average='micro')
recall = make_scorer(recall_score, average='micro')
acc=cross_val_score(clf, x, y, cv=10, scoring='accuracy').mean()
prec=cross_val_score(clf, x, y, cv=10, scoring=precision).mean()
rec=cross_val_score(clf, x, y, cv=10, scoring=recall).mean()
print("N-fold Cross Validation: accuracy = ",acc,', precision = ', prec, ' , recall = ',rec)


# In[ ]:




