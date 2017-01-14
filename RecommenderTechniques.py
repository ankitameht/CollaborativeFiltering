
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# In[2]:

#reading users file 
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u.user', sep='|', names=u_cols)


# In[3]:

#reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u.data', sep='\t', names=r_cols)


# In[5]:

print ratings.head()


# In[4]:

#reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u.item', sep='|', names=i_cols)


# In[5]:

n_u = ratings.user_id.unique().shape[0]
n_i = ratings.movie_id.unique().shape[0]
print 'Number of users = ' + str(n_u) + ' | Number of movies = ' + str(n_i)  


# In[6]:

rating_mat = np.zeros((n_u, n_i))
for line in ratings.itertuples():
    rating_mat[line[1]-1, line[2]-1] = line[3]  


# In[7]:

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u1.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('/Users/mukundverma/Downloads/ml-100k/u1.test', sep='\t', names=r_cols, encoding='latin-1')
print ratings_train.head()


# In[8]:

#creating user-item matrices :train and test
train_mat = np.zeros((n_u, n_i))
for line in ratings_train.itertuples():
    train_mat[line[1]-1, line[2]-1] = line[3]  

test_mat = np.zeros((n_u, n_i))
for line in ratings_test.itertuples():
    test_mat[line[1]-1, line[2]-1] = line[3]

print train_mat.shape, test_mat.shape


# In[92]:

print train_mat[:8,:8]


# In[162]:

from scipy.spatial.distance import cosine
u_similarity = np.zeros((n_u, n_u))
for i in range(u_similarity.shape[1]) :
    for j in range(u_similarity.shape[1]) :
      u_similarity[i,j] = 1-cosine(train_mat[i,:],train_mat[j,:])


# In[70]:

print u_similarity[:4,:4]
print u_similarity.shape


# In[136]:

i_similarity = np.zeros((n_i, n_i))
for i in range(i_similarity.shape[1]) :
    for j in range(i_similarity.shape[1]) :
      i_similarity[i,j] = 1-cosine(train_mat[:,i],train_mat[:,j])

print i_similarity[:4,:4]
print i_similarity.shape
    
 # nan: print i_similarity[400,1581   


# In[9]:

def userCorr(n_u):
    u_corr = np.zeros((n_u,n_u))
    u_corr = np.corrcoef(train_mat)
    return u_corr


# In[10]:

userCor = np.zeros((n_u,n_u))
userCor = userCorr(n_u)


# In[16]:

def itemCorr(n_i):
    train_copy = train_mat.T
    i_corr = np.zeros((n_i,n_i))
    print i_corr.shape
    i_corr = np.corrcoef(train_copy)
    return i_corr


# In[119]:

itemCor = np.zeros((n_i,n_i))
itemCor = itemCorr(n_i)


# In[120]:

u_prediction = predict(train_mat, userCor, kind='user')
i_prediction = predict(train_mat, itemCor, kind='item')
print 'UserUser NMAE' + str(errorMAE(u_prediction, test_mat)/4)
i_prediction[np.isnan(i_prediction)] = 0.0
test_mat[np.isnan(test_mat)] = 0.0
print 'ItemItem NMAE' + str(errorMAE(i_prediction, test_mat)/4)


# In[77]:

def topKUsers(uid,sim_mat,k=10):
    uRow = np.zeros((sim_mat.shape[1]))
    uRow = sim_mat[uid,:]
    neighbours = np.zeros((k+1))
    neighbours = np.argsort(uRow)[::-1][:k+1]
    return neighbours
        
topKUsers(273,u_similarity)


# In[72]:

i_similarity[np.isnan(i_similarity)] = 0.0


# In[78]:

def topKItems(iid,sim_mat,k=10):
    iRow = np.zeros((sim_mat.shape[1]))
    iRow = sim_mat[iid,:]
    neighbourItems = np.zeros((k+1))
    neighbourItems = np.argsort(iRow)[::-1][:k+1]
    return neighbourItems
        
topKItems(1272,i_similarity)


# In[20]:

def predictUserThreshold(mat,sim,t):
    prediction = np.zeros((mat.shape))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            prediction[i,j] = sim[i,:].dot(mat[:,j])/np.sum(np.abs(sim[i,:]))
    return prediction

def predictItemThreshold(mat,sim):
    prediction = np.zeros((mat.shape))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            prediction[i,j] = sim[j,:].dot(mat[i,:].T)/np.sum(np.abs(sim[j,:]))
    return prediction


# In[73]:

def predict(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


# In[74]:

u_prediction = predict(train_mat, u_similarity, kind='user')


# In[75]:

i_prediction = predict(train_mat, i_similarity, kind='item')


# In[12]:

from sklearn.metrics import mean_squared_error
from math import sqrt
def error(predvalue, labels):
    predvalue = predvalue[labels.nonzero()].flatten() 
    labels = labels[labels.nonzero()].flatten()
    return sqrt(mean_squared_error(predvalue, labels))


# In[11]:

from sklearn.metrics import mean_absolute_error
def errorMAE(predvalue, labels):
    predvalue = predvalue[labels.nonzero()].flatten() 
    labels = labels[labels.nonzero()].flatten()
    return mean_absolute_error(predvalue, labels)


# In[76]:

print 'User-based CF MSE: ' + str(errorMAE(u_prediction, test_mat))
i_prediction[np.isnan(i_prediction)] = 0.0
test_mat[np.isnan(test_mat)] = 0.0
print 'Item-based CF MSE: ' + str(errorMAE(i_prediction, test_mat))


# In[77]:

print 'UserUser NMAE' + str(errorMAE(u_prediction, test_mat)/4)
print 'ItemItem NMAE' + str(errorMAE(i_prediction, test_mat)/4)



# In[92]:

'''def predictTopkUser(mat, sim, k=10):
    prediction = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        k_users = [np.argsort(sim[:,i])[:-k-1:-1]]
        for j in range(mat.shape[1]):
            prediction[i, j] = sim[i, :][k_users].dot(ratings[:, j][k_users]) 
            prediction[i, j] /= np.sum(np.abs(sim[i, :][k_users]))
    return prediction
   
def predictTopkItem(mat, sim, k=10):
    prediction = np.zeros(mat.shape)
    for j in range(mat.shape[1]):
        k_items = [np.argsort(sim[:,j])[:-k-1:-1]]
        for i in range(mat.shape[0]):
            prediction[i, j] = sim[j, :][k_items].dot(ratings[i, :][k_items].T) 
            prediction[i, j] /= np.sum(np.abs(sim[j, :][k_items]))        
    return prediction'''
def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred


# In[95]:

pred = predict_topk(train_mat, u_similarity, kind='user', k=40)
print 'Top-k User-based CF MSE: ' + str(error(pred, test_mat))

pred = predict_topk(train_mat, i_similarity, kind='item', k=40)
pred[np.isnan(pred)] = 0.0
print 'Top-k Item-based CF MSE: ' + str(error(pred, test_mat))


# In[12]:

#significance weighting with neighbour selection
def sigWeightUser(ratings, similarity, n_user, k):
    pred = np.zeros(ratings.shape)
    newSim = np.zeros((n_user, n_user))
    corated = np.zeros((n_user, n_user))
    count = 0
    for i in range(corated.shape[0]) :
        for j in range(corated.shape[0]) :
            for k in range(ratings.shape[1]):
                if ratings[i,k] != 0 and ratings[j,k] != 0:
                    corated [i,j] = corated[i,j] + 1
                
    for i in range(newSim.shape[0]) :
        for j in range(newSim.shape[0]) :
            if corated[i,j] < 50:
                newSim[i,j] = similarity[i,j] * (corated[i,j]/50)
    for i in range(ratings.shape[0]):
        top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
        for j in range(ratings.shape[1]):
            
            pred[i, j] = newSim[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
            pred[i, j] /= np.sum(np.abs(newSim[i, :][top_k_users]))
    return pred
    


# In[13]:

predSig = sigWeightUser(train_mat, userCor,n_u,k=50)


# In[167]:

predSig[np.isnan(predSig)] = 0.0
print 'Top-k User-based significance weighting MSE: ' + str(errorMAE(predSig, test_mat)/4)


# In[147]:

#Variance weighting 
def varianceWeightUser(ratings, similarity, n_user):
    pred = np.zeros(ratings.shape)
    userVar = np.zeros((n_user))
    for i in range(n_user):
        userVar[i] = np.var(ratings[:,i])
    print userVar.shape
    maxVar = np.max(userVar)
    minVar = np.min(userVar)
    for i in range(ratings.shape[0]):
      for j in range(ratings.shape[1]):
          mFactor = (userVar[j] - minVar)/maxVar
          pred[i,j] = similarity[i, :].dot(ratings[:, j] * mFactor) 
          pred[i,j] /= np.sum(np.abs(similarity[i, :]))
    return pred


# In[163]:

predVariance = varianceWeightUser(train_mat, u_similarity, n_i)


# In[164]:

predVariance[np.isnan(predVariance)] = 0.0
print 'Top-k User-based variance weighting MSE: ' + str(errorMAE(predVariance, test_mat)/4)


# In[ ]:



