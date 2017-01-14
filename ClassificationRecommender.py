
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import distance


# In[2]:

#reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u.data', sep='\t', names=r_cols)


# In[3]:

n_u = ratings.user_id.unique().shape[0]
n_i = ratings.movie_id.unique().shape[0]
print 'Number of users = ' + str(n_u) + ' | Number of movies = ' + str(n_i)  


# In[4]:

rating_mat = np.zeros((n_u, n_i))
for line in ratings.itertuples():
    rating_mat[line[1]-1, line[2]-1] = line[3]  


# In[5]:

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u1.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u1.test', sep='\t', names=r_cols, encoding='latin-1')
print ratings_train.head()


# In[6]:

#creating user-item matrices :train and test
train_mat = np.zeros((n_u, n_i))
for line in ratings_train.itertuples():
    train_mat[line[1]-1, line[2]-1] = line[3]  

test_mat = np.zeros((n_u, n_i))
for line in ratings_test.itertuples():
    test_mat[line[1]-1, line[2]-1] = line[3]


# In[7]:

train_mat = train_mat.T


# In[8]:

test_mat = test_mat.T


# In[9]:

#binarizing the values
for i in range(train_mat.shape[0]) :
    for j in range(train_mat.shape[1]) :
        if train_mat[i,j] != 0:
            if train_mat[i,j] < 3:
                train_mat[i,j] = -1
            else:
                train_mat[i,j] = 1
for i in range(test_mat.shape[0]) :
    for j in range(test_mat.shape[1]) :
        if test_mat[i,j] != 0:
            if test_mat[i,j] < 3:
                test_mat[i,j] = -1
            else:
                test_mat[i,j] = 1            
            
print train_mat.shape, test_mat.shape
print train_mat[1][5], test_mat[1][5]
            


# In[10]:

#counting the number of zeroes in a matrix
c = 0
for i in range(test_mat.shape[0]) :
    for j in range(test_mat.shape[1]) :
        if test_mat[i,j] == 0:
            c += 1
print c            


# In[84]:

#nearest neighbour classifier
def kNN(train,test):
    hammingDist = np.zeros((train.shape[0]))
    minDist = []
    for j in range(test.shape[1]):
        for i in range(test.shape[0]):
            if test[i][j] == 0:
                row1 = test[i]
                hammingDist[i] = 10000
                for k in range(train.shape[0]):
                    if k != i:
                        row2 = train[k]
                        hammingDist[k] = len([a for a, b in zip(row1, row2) if a == b and a!=0 and b!=0])   
                minDist = np.where(hammingDist == hammingDist.min())[0]
                #print minDist
                for item in minDist:
                    
                    #print train[item][j]
                    if train[item][j]!= 0: 
                        test[i][j] =  train[item][j]
                        break
            
            
    return test    


# In[ ]:

predicted = kNN(train_mat,test_mat)


# In[ ]:

ratings_global = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u1.base', sep='\t', names=r_cols, encoding='latin-1')
global_mat = np.zeros((n_u, n_i))
for line in ratings_global.itertuples():
    global_mat[line[1]-1, line[2]-1] = line[3]


# In[ ]:

predict = predicted[predicted.nonzero()].flatten()
globalRating = global_mat[predicted.nonzero()].flatten()
error =  np.count_nonzero(predicted==user)/float(np.count_nonzero(predicted))
print error



# In[66]:

#Naive Bayes classifier
##global priors
n1 = 0  #positive class
n2 = 0  #negative class
for i in range(train.shape[0]) :
    for j in range(train.shape[1]) :
        if train[i,j] != 0:
            if train[i,j] == 1:
                n1 += 1
            else:
                n2 += 1
print n1,n2           


# In[67]:

total = train.shape[0] * train.shape[1]
print total
print n1
n1 =  n1/float(total)
n2 = n2/float(total)
print n1, n2


# In[80]:

def nb(train):
    cp1 = 0
    cp2 = 0
    posRow = []
    pred = np.zeros((train.shape[0],train.shape[1]))
    for j in range(train.shape[1]):
        label = train[:,j]
        cp1 = np.count_nonzero(label == 1)
        cp2 = np.count_nonzero(label == -1)
        for i in range(train.shape[0]):
            if train[i][j] == 0:
                row1 = train[i]  #feature for missing user
                c1Row = np.where(label == 1)[0]
                c2Row = np.where(label == -1)[0]
                
                
                
                


# In[78]:

t = np.where(train[:,3] == 0)[0]
print t[1]


# In[83]:

ratings_train.shape


# In[ ]:



