from PyReco import SVDRecommender, CFRecommender
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(y_t, y_p):
       print('RMSE:\t', np.sqrt(mean_squared_error(y_t, y_p)))
       print('MAE:\t', mean_absolute_error(y_t, y_p))

os.chdir('ml-latest-small')
data = pd.read_csv('ratings.csv')
user_item_data = data.pivot_table(values = 'rating', index = 'userId', columns = 'movieId')
train, test = train_test_split(data, test_size = 0.1, random_state = 69)
train = train.pivot_table(values = 'rating', index = 'userId', columns = 'movieId')
test.reset_index(drop = True, inplace = True)

recommender = CFRecommender('ubcf', verbose = True)
recommender.fit(train, fill = 'mean', sim_engine = 'cosine') # fill = {'mean', 'zero', 'interpolate'}, sim_engine = {'cosine', 'euclidean', 'pearson'}
recommender.predict(212, 1320, method = 'weighted') # method = {'mean', 'weighted'}
recommender.reco(212)
recommender.top_matches(212)

for i in range(1, 11):
       print(i)
       recommender = SVDRecommender(verbose = True)
       recommender.fit(train, method = 'fa', k = i)
       y_pred = list(test[['userId', 'movieId']].apply(lambda row: recommender.predict(row[0], row[1]), axis = 1))
       y_true = test['rating']
       evaluate(y_true, y_pred)
       print()

"""
OPTIMAL PARAMETERS
SVDRecommender:
{method: 'slice', k = 9,
method: 'pca', k = 1,
method: 'fa', k = NA}
Best performance: RMSE = 0.92759, MAE = 0.71361


"""