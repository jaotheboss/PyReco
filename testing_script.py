from PyReco import SVDRecommender, CFRecommender, read_timer
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

# Evaluation
recommender = CFRecommender('ubcf', verbose = False)
recommender.fit(train, fill = 'zero', sim_engine = 'cosine') # fill = {'mean', 'zero', 'interpolate'}, sim_engine = {'cosine', 'euclidean', 'pearson'}
y_pred = list(test[['userId', 'movieId']].apply(lambda row: recommender.predict(row[0], row[1], method = 'mean'), axis = 1))
y_true = test['rating']
evaluate(y_true, y_pred)

# Grid-Search Evaluation
parameters = {
       'rec_fill': ['mean', 'zero', 'interpolate'],
       'rec_sim_engine': ['cosine', 'euclidean', 'pearson'],
       
       'pred_method': ['mean', 'weighted']
}
import time
for pred_method in parameters['pred_method']:
       for rec_sim_engine in parameters['rec_sim_engine']:
              for rec_fill in parameters['rec_fill']:
                     print('Prediction Method:',pred_method, '\nSimilarity Engine:', rec_sim_engine, '\nFitting Fill Method:', rec_fill)
                     start_time = time.time()
                     recommender = CFRecommender('ubcf', verbose = False)
                     recommender.fit(train, fill = rec_fill, sim_engine = rec_sim_engine)
                     y_pred = list(test[['userId', 'movieId']].apply(lambda row: recommender.predict(row[0], row[1], method = pred_method), axis = 1))
                     y_true = test['rating']
                     evaluate(y_true, y_pred)
                     end_time = time.time()
                     print(read_timer(end_time - start_time), '\n')
