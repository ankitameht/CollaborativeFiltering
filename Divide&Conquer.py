
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from fancyimpute import MatrixFactorization
import pickle

from sklearn.metrics import mean_absolute_error
def errorMAE(predvalue, labels):
    predvalue = predvalue[labels.nonzero()].flatten() 
    labels = labels[labels.nonzero()].flatten()
    return mean_absolute_error(predvalue, labels)

# In[2]:

ratings = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-10M/ratings.dat', sep='::', header=None)

n_u = ratings[0].unique().shape[0]
n_i = ratings[1].unique().shape[0]


# In[26]:

ratings[2].unique()


# In[3]:

msk = np.random.rand(10000054) < 0.8
train = ratings[msk]
test = ratings[~msk]


# In[4]:

rating_mat = np.zeros((train.shape[0], train.shape[1]))
unique_user = []
unique_item = []
unique_user = ratings[0].unique()
unique_item = ratings[1].unique()
print unique_user
u_dict = dict()
i_dict = dict()

    


# In[5]:

for i in range(0,len(unique_user)):
    u_dict[unique_user[i]] = i
for i in range(0,len(unique_item)):
    i_dict[unique_item[i]] = i


# In[7]:

train.shape


# In[6]:

train_mat = np.zeros((n_u,n_i))
for line in train.itertuples():
    train_mat[u_dict[line[1]], i_dict[line[2]]] = line[3]


# In[7]:

test_mat = np.zeros((n_u,n_i))
for line in test.itertuples():
    test_mat[u_dict[line[1]], i_dict[line[2]]] = line[3]


# In[ ]:


pickle.dump(train_mat, open('trainMatrix.dat', 'wb'))


# In[8]:

np.save('trainMatrix.npy', train_mat)


# In[12]:

np.save('testMatrix.npy', test_mat)


# In[15]:

c1 = np.zeros((n_u,n_i/11))


# In[20]:

for i in range(0,c1.shape[0]):
    for j in range(0,c1.shape[1]):
        c1[i,j] = train_mat[i,j]


# In[21]:

np.save('c1.npy',c1)


# In[22]:

c1[c1==0] = np.nan


# In[23]:

from fancyimpute import SoftImpute
completedMat=SoftImpute(shrinkage_value=40,max_iters=35).complete(c1)


# In[24]:

np.save("partition1.npy",completedMat)


# In[25]:

print completedMat[0:5][0:5]


# In[ ]:

U, s, V = np.linalg.svd(completedMat, full_matrices=True)


# In[ ]:

L_proj = U.dot(U.T).dot(train_mat)
 ## Repeated the same code 11 times to calculate NMAE = 0.3354277822
