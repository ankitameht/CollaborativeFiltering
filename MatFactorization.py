
# coding: utf-8

# In[1]:

import pandas as pd

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')


# In[3]:

print items.shape
items.head()


# In[24]:

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u2.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u2.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_base.shape, ratings_test.shape


# In[25]:

import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)


# In[4]:

train_data.show()


# In[6]:

#factorization_recommender.create
popularity_model = graphlab.factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')


# In[7]:

popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)


# In[8]:

popularity_model['coefficients']


# In[9]:

test_data.head()


# In[12]:

loaded_model=popularity_model

#load the test data set
test_set=test_data
actual_ratings=test_set.select_column('rating')

#predict the ratings on test data set
predicted_ratings=loaded_model.predict(test_set)
predicted_ratings.save('predictions_FM')

#Calculate RMSE for predictions
if (actual_ratings.size()!=predicted_ratings.size()):
    print("There exists a mismatch in number of predictions and actuals")
else:
    rmse_val=graphlab.evaluation.rmse(actual_ratings,predicted_ratings)
    print("RMSE=\n")
    print(rmse_val)

#print the summary for the model
print(loaded_model.summary())



# In[13]:

predicted_ratings


# In[14]:

predicted_ratings.shape


# In[4]:

from sklearn.metrics import mean_absolute_error
def errorMAE(predvalue, labels):
    return mean_absolute_error(predvalue, labels)


# In[19]:

print 'NMAE: ' + str(errorMAE(predicted_ratings, actual_ratings)/4)


# In[38]:

model2 = graphlab.factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', num_factors=150, max_iterations=3500)


# In[39]:

loaded_model=model2

#load the test data set
test_set=test_data
actual_ratings=test_set.select_column('rating')

#predict the ratings on test data set
predicted_ratings=loaded_model.predict(test_set)
#predicted_ratings.save('predictions_FM')

#Calculate RMSE for predictions
if (actual_ratings.size()!=predicted_ratings.size()):
    print("There exists a mismatch in number of predictions and actuals")
else:
    rmse_val=graphlab.evaluation.rmse(actual_ratings,predicted_ratings)
    print("RMSE=\n")
    print(rmse_val)

#print the summary for the model
#print(loaded_model.summary())




# In[40]:

print 'NMAE: ' + str(errorMAE(predicted_ratings, actual_ratings)/4)


# In[ ]:

#20 40 80 100 200

