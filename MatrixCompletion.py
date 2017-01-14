import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from fancyimpute import SoftImpute

def nmae(test,prediction):
    prediction = prediction[test.nonzero()].flatten()
    test = test[test.nonzero()].flatten()
    return mean_absolute_error(prediction, test)/4

header = ['uid', 'iid', 'rating', 'timestamp']
data_global = pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u.data', sep='\t', names=header)

for i in range(1,6):
    data=pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u'+str(i)+'.base', sep='\t', names=header)
    user_item=np.zeros((data_global.uid.unique().shape[0],data_global.iid.unique().shape[0]))
    for row in data.itertuples():
        user_item[row[1]-1, row[2]-1] = row[3]
    test_data=pd.read_csv('/Users/mukundverma/Desktop/Courses/Collaborative Filtering/ml-100k/u'+str(i)+'.test', sep='\t', names=header)
    test=np.zeros((data_global.uid.unique().shape[0],data_global.iid.unique().shape[0]))
    for row in test_data.itertuples():
        test[row[1]-1, row[2]-1] = row[3]

    user_item1=user_item
    user_item1[user_item1==0]=np.nan
    newR=SoftImpute(shrinkage_value=150,max_iters=100).complete(user_item1)
    print nmae(test,newR)
