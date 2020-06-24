"""
2 kinds of recommendation engines:

Engine 1 = Collaborative Filtering

1. User-Based Collaborative Filtering
2. Item-Based Collaborative Filtering

1. Compare users that have similar patterns. Recommend product to either based
on the fact that similar people that are of the same kind also use that product.
_DATASET: This requires pattern data of each user with relation to the item
itself. For example, col_name = [user_id, product1, product2, ..., ] and each
product could consist of whether or not the user used/likes the product

2. For each product, look at its user base. Based on those that use this particular
product, look at what is the next most used product among this user base. You can
then recommend this particular product to those that are not using (from within the
user base)
_DATASET: Requires the same dataset as above.

Engine 2 = Content Based

Content based filtering looks at products and groups them based on their
similarity. Top rated products, products that are similar and etc.
_DATASET: dataset of the product itself and their portfolio

References:
https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
https://www.geeksforgeeks.org/python-implementation-of-movie-recommender-system/
"""

# needed for both
import time
import numpy as np                                                    # handling of arrays

# needed for CFRecommender
from scipy.spatial.distance import cosine, euclidean                  # vector similarity evaluaters
from sklearn.metrics.pairwise import paired_euclidean_distances, cosine_similarity
from scipy.stats import pearsonr
from itertools import compress

# needed for SVDRecommender
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA, FactorAnalysis

# defining helper functions
def read_timer(seconds):
       """
       Convert the timers readings into something more legible
       """
       minutes = int(seconds // 60)
       hours = int(minutes // 60)
       seconds = int(seconds % 60)
       return "Elapsed Time: {h} hours, {m} minutes and {s} seconds.".format(h = hours, m = minutes, s = seconds)
       
# defining main recommendation engine/model classes
class CFRecommender():
       """
       A recommender model that is based on collaborative filtering. This means
       that the model predicts the the users' rating of the object based on the 
       similarities between 1. users (UBCF) or 2. items (ICBF)
       """
       def __init__(self, method, verbose = False):
              self.method = method        # ubcf, ibcf
              self.verbose = verbose
              self.similarity_matrix = None
              self.items = []
              self.item_means = 0
              self.users = []
              self.user_means = 0
              self.data = None

       def fit(self, user_item_matrix, fill = 'mean', sim_engine = 'cosine'):
              """
              Performs an algorithm that will create a similarity matrix of each
              user with each other. 

              Parameters:
                     user_item_matrix : pandas dataframe
                            Users as rows and Items as columns.
                            The index names should be the user_ids while the column
                            names should be the item_ids

                     fill : string, default = 'mean', {'mean', 'zero', 'interpolate'}
                            Chooses how to fill the nan values in the data.
                            Note that only linear interpolation is available

                     sim_engine : string, default = 'cosine', {'cosine', 'euclidean', 'pearson'}
                            The engine/algorithm that will be used to calculate the
                            similarity between each user/item

              Returns:
                     Nothing. The function updates the similarity matrix attribute
              """
              # start timer
              if self.verbose:
                     start_time = time.time()
              # update user and item list for fitted data
              self.users, self.items = list(user_item_matrix.index), list(user_item_matrix.columns)
              self.data = user_item_matrix

              # filling the nan's
              if self.verbose:
                     print('Reconciling data...')
              if fill == 'mean':
                     user_item_matrix = np.ma.masked_array(user_item_matrix, np.isnan(np.array(user_item_matrix)))
                     self.item_means = np.mean(user_item_matrix, axis = 0)
                     self.user_means = np.mean(user_item_matrix, axis = 1)
                     user_item_matrix = user_item_matrix.filled(self.item_means)

              elif fill == 'zero':
                     user_item_matrix = np.array(user_item_matrix.fillna(0))
              
              elif fill == 'interpolate':
                     user_item_matrix = np.array(user_item_matrix.interpolate(limit_direction = 'both'))

              else:
                     raise AttributeError('Fill method not recognised. Call for either mask or zero.')
              
              # creating the similarity matrix
              if self.method == 'ibcf':
                     user_item_matrix = user_item_matrix.T
              if self.verbose:
                     print('Forming the similarity matrix...')
              if sim_engine == 'cosine':
                     first_user_vector = user_item_matrix[0]
                     similarity_matrix = np.apply_along_axis(lambda x: cosine_similarity([first_user_vector], [x])[0][0], 1, user_item_matrix)
                     if self.method == 'ubcf':
                            for i in range(1, len(self.users)):
                                   user_vector = user_item_matrix[i]
                                   similarity_matrix = np.vstack((similarity_matrix, np.apply_along_axis(lambda x: cosine_similarity([user_vector], [x])[0][0], 1, user_item_matrix)))
                     elif self.method == 'ibcf':
                            for i in range(1, len(self.items)):
                                   user_vector = user_item_matrix[i]
                                   similarity_matrix = np.vstack((similarity_matrix, np.apply_along_axis(lambda x: cosine_similarity([user_vector], [x])[0][0], 1, user_item_matrix)))
                     else:
                            raise AttributeError('Method not recognised. Call for either ubcf or ibcf')

              elif sim_engine == 'euclidean':
                     first_user_vector = user_item_matrix[0]
                     similarity_matrix = np.apply_along_axis(lambda x: -paired_euclidean_distances([first_user_vector], [x])[0][0], 1, user_item_matrix)
                     if self.method == 'ubcf':
                            for i in range(1, len(self.users)):
                                   user_vector = user_item_matrix[i]
                                   similarity_matrix = np.vstack((similarity_matrix, np.apply_along_axis(lambda x: -paired_euclidean_distances([user_vector], [x])[0], 1, user_item_matrix)))
                     elif self.method == 'ibcf':
                            for i in range(1, len(self.items)):
                                   user_vector = user_item_matrix[i]
                                   similarity_matrix = np.vstack((similarity_matrix, np.apply_along_axis(lambda x: -paired_euclidean_distances([user_vector], [x])[0], 1, user_item_matrix)))
                     else:
                            raise AttributeError('Method not recognised. Call for either ubcf or ibcf')

              elif sim_engine == 'pearson':
                     first_user_vector = user_item_matrix[0]
                     similarity_matrix = np.apply_along_axis(lambda x: jaccard_score(first_user_vector, x)[0], 1, user_item_matrix)
                     if self.method == 'ubcf':
                            for i in range(1, len(self.users)):
                                   user_vector = user_item_matrix[i]
                                   similarity_matrix = np.vstack((similarity_matrix, np.apply_along_axis(lambda x: pearsonr(user_vector, x)[0], 1, user_item_matrix)))
                     elif self.method == 'ibcf':
                            for i in range(1, len(self.items)):
                                   user_vector = user_item_matrix[i]
                                   similarity_matrix = np.vstack((similarity_matrix, np.apply_along_axis(lambda x: pearsonr(user_vector, x)[0], 1, user_item_matrix)))
                     else:
                            raise AttributeError('Method not recognised. Call for either ubcf or ibcf')

              else:
                     raise AttributeError('Sim engine attribute not recognised. Call for either cosine, euclidean or pearson.')
              
              self.similarity_matrix = similarity_matrix
              if self.verbose:
                     print('Model fitted!')
                     stop_time = time.time()
                     print(read_timer(stop_time - start_time))

       def predict(self, user, item, method = 'mean', k = 10):
              """
              Predicts the users' rating for the item

              Parameters:
                     user : string or integer
                            The value you used to label each user

                     item : string or integer   
                            The value you used to label each item

                     method : string, default = 'mean', {'mean', 'weighted'}
                            The method in which the algorithm will merge the
                            scores of the users/items that have been filtered
                     
                     k : integer, default = 10
                            The number of users/item to collaboratively filter.
                            ie. the top k similar users/items
              
              Returns:
                     output : integer
                            The prediction of the users' rating for the 
                            particular item
              """
              # grabbing indexes of the user and item
              user_index = self.users.index(user)
              item_index = self.items.index(item)
              curr_value = np.array(self.data)[user_index, item_index]

              # if there already exist a rating, return it
              if ~np.isnan(curr_value):
                     return curr_value

              # if user-based collaborative filtering is selected
              elif self.method == 'ubcf':
                     # finding similar users
                     if self.verbose:
                            print('Sieving through similar users...')
                     useritem_simvector = self.similarity_matrix[user_index]
                     sort_index = list(np.argsort(useritem_simvector)[::-1])
                     sort_index.remove(user_index)

                     # retrieving the ratings of the item from users that are similar
                     if self.verbose:
                            print('Processing prediction...')
                     # taking the first 10
                     rate_vector = [np.array(self.data)[sort_index[i], item_index] for i in range(k)]

                     if sum(~np.isnan(rate_vector)) < 5:
                            # if there's less than 5 similar user ratings, take the first 10 valid users
                            print('Top', k, 'users/items had < 5 ratings. Retrying with top', k, 'valid users/items.')
                            rate_vector = []
                            i = 0
                            while len(rate_vector) < k:
                                   value = np.array(self.data)[sort_index[i], item_index]
                                   if ~np.isnan(value):
                                          rate_vector.append(value)
                                   i += 1

                     # combine all the ratings either via mean or weighted methods
                     if method == 'mean':
                            return np.nanmean(rate_vector)

                     elif method == 'weighted':
                            index_mask = ~np.isnan(rate_vector)
                            return np.average(list(compress(rate_vector, index_mask)), weights = list(compress(range(k, 0, -1), index_mask)))
                     else:
                            raise AttributeError('Method not recognised. Call for mean or weighted.')

              # do the same as ubcf just for items instead of users
              elif self.method == 'ibcf':
                     if self.verbose:
                            print('Sieving through similar items...')
                     useritem_simvector = self.similarity_matrix[item_index]
                     sort_index = list(np.argsort(useritem_simvector)[::-1])
                     sort_index.remove(item_index)

                     if self.verbose:
                            print('Processing prediction...')
                     # taking the first 10
                     rate_vector = [np.array(self.data)[sort_index[i], item_index] for i in range(k)]

                     if sum(~np.isnan(rate_vector)) < 5:
                            # if there's less than 5 similar user ratings, take the first 10 valid users
                            print('Top', k, 'users/items had < 5 ratings. Retrying with top', k, 'valid users/items.')
                            rate_vector = []
                            i = 0
                            while len(rate_vector) < k:
                                   value = np.array(self.data)[sort_index[i], item_index]
                                   if ~np.isnan(value):
                                          rate_vector.append(value)
                                   i += 1

                     if method == 'mean':
                            return np.nanmean(rate_vector)

                     elif method == 'weighted':
                            index_mask = ~np.isnan(rate_vector)
                            return np.average(list(compress(rate_vector, index_mask)), weights = list(compress(range(k, 0, -1), index_mask)))
                     else:
                            raise AttributeError('Method not recognised. Call for mean or weighted.')
              else:
                     raise AttributeError('Collaborative filtering method not recognised. Call for either ubcf or ibcf')
       
       def reco(self, user, method = 'mean', k = 10, show = 10):
              """
              Recommends all the items to the user based on the k most similar
              items/users

              Parameters:
                     user : string or integer
                            The value that you used to label the user

                     method : string, default = 'mean', {'mean', 'weighted'}
                            The method in which the algorithm will merge the
                            scores of the users/items that have been filtered

                     k : integer, default = 10
                            The number of users/items to look at to recommend 
                            items to the user

                     show : integer, default = 10
                            Number of recommendations to show

              Returns:
                     output : list
                            A list containing k + 1 elements of the top recommendations
              """
              # check user validity
              if user not in self.users:
                     raise ValueError('No such user in fitted database. Please try anoter user')
              user_index = self.users.index(user)
              newitems_mask = np.isnan(np.array(self.data)[user_index])

              # extracting the top k most similar items or users
              if self.method == 'ubcf':
                     if self.verbose:
                            print('Sieving through similar users...')
                     useritem_simvector = self.similarity_matrix[user_index]
                     sort_index = list(np.argsort(useritem_simvector)[::-1])
                     sort_index.remove(user_index)
                     
                     if self.verbose:
                            print('Preparing recommended items and ratings..')
                     similar_users_ratings = np.array(self.data)[sort_index[:k],]
                     mean_simusers_ratings = np.nanmean(similar_users_ratings, axis = 0)
                     reco_items = list(compress(list(enumerate(mean_simusers_ratings)), newitems_mask))
                     if len(reco_items) == 0:
                            return []
                     else:
                            reco_items = [i for i in reco_items if ~np.isnan(i[1])]
                            reco_items.sort(key = lambda x: x[1], reverse = True)
                            result = [['Item', 'Rating']] + [[self.items[i], j] for i, j in reco_items[:show]]
                            return result

              elif self.method == 'ibcf':
                     if self.verbose:
                            print('Sieving through similar users...')
                     new_items_index = [i for i, j in enumerate(newitems_mask) if j]
                     result = []
                     if self.verbose:
                            print('Preparing recommended items and ratings..')
                     for i in new_items_index:
                            item = self.items[i]
                            rating = self.predict(user, item, 'weighted', k)
                            result.append([item, rating])
                     result.sort(key = lambda x: x[1], reverse = True)
                     result = [['Item', 'Rating']] + result[:show]
                     return result
              else:
                     raise AttributeError('Collaborative filtering method not recognised. Call for ubcf or ibcf.')

              # if self.last_user_reco['user'] != user:          # apply memoisation
              #        if self.verbose:
              #               print('Crafting ratings vector...')
              #        ratings_vector = [self.predict(user, i, method, k) for i in self.items]
              #        self.last_user_reco['user'] = user
              #        self.last_user_reco['vector'] = ratings_vector
              # else:
              #        ratings_vector = self.last_user_reco['vector']
              
              # newitem_mask_index = np.isnan(np.array(self.data)[user_index])
              # result = [['Item', 'Rating']]
              # if self.verbose:
              #        print('Creating the recommendations list...')
              # if repeat:
              #        #item_rating = list(zip(self.items, ratings_vector)).sort(key = lambda x: x[1], reverse = True)
              #        #for i in range(k):
              #        #       result.append(list(item_rating[i]))
              #        sort_index = np.argsort(ratings_vector)[::-1]
              #        for i in range(k):
              #               rating = ratings_vector[sort_index[i]]
              #               item = self.items[sort_index[i]]
              #               result.append([item, rating])
              #        return result
              # else:
              #        ratings_vector = ratings_vector[newitem_mask_index]
              #        item_list = self.items[newitem_mask_index]
              #        #item_rating = list(zip(item_list, ratings_vector)).sort(key = lambda x: x[1], reverse = True)
              #        #for i in range(k):
              #        #       result.append(list(item_rating[i]))
              #        sort_index = np.argsort(ratings_vector)[::-1]
              #        for i in range(k):
              #               rating = ratings_vector[sort_index[i]]
              #               item = item_list[sort_index[i]]
              #               result.append([item, rating])
              #        return result
       
       def top_matches(self, useritem, k = 3):
              """
              Returns the top k most similar items/users from the input

              Parameters:
                     useritem : string or integer
                            The user or item that you want to find the similar 
                            objects from

                     k : integer, default = 3
                            The number of similar objects you want to see from the 
                            input
              """
              if self.method == 'ubcf':
                     user_index = self.users.index(useritem)

                     useritem_simvector = self.similarity_matrix[user_index]
                     sort_index = list(np.argsort(useritem_simvector)[::-1])
                     sort_index.remove(user_index)

                     result = [['User', 'Similarity Score']]
                     for i in range(k):
                            result.append([self.users[sort_index[i]], useritem_simvector[sort_index[i]]])
                     return result

              elif self.method == 'ibcf':
                     item_index = self.items.index(useritem)

                     useritem_simvector = self.similarity_matrix[item_index]
                     sort_index = list(np.argsort(useritem_simvector)[::-1])
                     sort_index.remove(item_index)

                     result = [['Item', 'Similarity Score']]
                     for i in range(k):
                            result.append([self.items[sort_index[i]], useritem_simvector[sort_index[i]]])
                     return result
              else:
                     raise AttributeError('Collaborative filtering method not recognised. Call for ubcf or ibcf.')

class SVDRecommender():
       """
       A recommender model that is based on matrix decomposition. This means 
       that the model will apply an algorithm to shrink and summarise the 
       matrix into latent factors. Thereafter, the model will use these
       factors to predict a users' rating of an object
       """
       def __init__(self, verbose = False):       
              self.verbose = verbose
              self.utility_matrix = None
              self.users = []
              self.items = []
              self.user_means = None
              self.item_means = None
              self.user_history = {}
              self.data = None

       def fit(self, user_item_matrix, k = 10, method = 'slice'):
              """
              Performs the Single Value Decomposition (SVD) on the user-item-matrix

              Parameters:
                     user_item_matrix : pandas dataframe
                            Users as rows and Items as columns.
                            The index names should be the user_ids while the column
                            names should be the item_ids
                     
                     k : integer, default = 4
                            The number of features in each latent factor. Basically, 
                            the number of dimensions you want to represent each user
                            and each item

                            Must not be more than min(number of items, number of users)

                     method : string, {'slice', 'pca', 'fa'}
                            The method in which is used to extract the k features from 
                            the resultant SVD matrix

                            IF 'slice' :  
                                   the SVD matrix will just be sliced to fit the k 
                                   dimension
                            IF 'pca' :
                                   principal component analysis (PCA) will be used 
                                   to summarise the whole matrix into k features
                            IF 'fa' : 
                                   factor analysis (FA) will be used to summarise 
                                   the whole matrix into k features
                     
              Returns:
                     Nothing. This function updates the utility matrix attribute
              """
              # start timer
              if self.verbose:
                     start_time = time.time()
              # update user and item list for fitted data
              self.users, self.items = list(user_item_matrix.index), list(user_item_matrix.columns)
              self.data = user_item_matrix

              # extract users current rating history
              if self.user_history == {}:
                     if self.verbose:
                            print('Processing users history database')
                     for user in self.users:
                            user_items = user_item_matrix.loc[[user]].values[0]
                            self.user_history[user] = [index for index, item in enumerate(user_items) if ~np.isnan(item)]

              # we start by masking the nan entries of the matrix to find item means
              user_item_matrix = np.array(user_item_matrix)
              masked_arr = np.ma.masked_array(user_item_matrix, np.isnan(user_item_matrix))
              self.item_means = np.mean(masked_arr, axis = 0)
              self.user_means = np.mean(masked_arr, axis = 1)

              # we now replace the nan's of each item with their means and center them at the mean
              utility_mat = masked_arr.filled(self.item_means)
              item_means_matrix = np.tile(self.item_means, (utility_mat.shape[0], 1))
              utility_mat = utility_mat - item_means_matrix

              # apply the SVD on the utility matrix
              if self.verbose:
                     print('Performing decomposition...')
              U, s, V = np.linalg.svd(
                     utility_mat, 
                     full_matrices = False
              )
              s = np.diag(s)

              # extract only k latent features from the decomposition
              if self.verbose:
                     print('Performing latent factorisation...')
              if len(self.items) < k:
                     raise ValueError('Make sure k <= min(number of items, number of users)')
              if method == 'slice':
                     U = U[:, 0:k]
                     s = s[0:k, 0:k]
                     V = V[0:k, :]
              elif method == 'pca':
                     pca = PCA(n_components = k)
                     U = pca.fit_transform(U)
                     s = pca.fit_transform(pca.fit_transform(s).T).T
                     V = pca.fit_transform(V.T).T
              elif method == 'fa':
                     fa = FactorAnalysis(n_components = k, random_state = 69)
                     U = fa.fit_transform(U)
                     s = fa.fit_transform(fa.fit_transform(s).T).T
                     V = fa.fit_transform(V.T).T
              else:
                     raise AttributeError('Decomposition method not recognized. Call for either slice, pca or fa')
              
              # form the utility matrix with the resulting arrays
              s_root = sqrtm(s)
              Usk = np.dot(U, s_root)
              skV = np.dot(s_root, V)
              utility_mat = np.dot(Usk, skV)
              utility_mat = utility_mat + item_means_matrix
              if method == 'pca' or method == 'fa':
                     utility_mat = np.round(utility_mat.real, 5)
              self.utility_matrix = utility_mat
              if self.verbose:
                     print('Model fitted!')
                     stop_time = time.time()
                     print(read_timer(stop_time - start_time))

       def predict(self, user, item):
              """
              Predicts the users' rating for the item

              Parameters:
                     user : string or integer
                            The value you used to label each user

                     item : string or integer   
                            The value you used to label each item

              Returns:
                     output : integer
                            The prediction of the users' rating for the 
                            particular item
              """
              # user and item in the test set may not always occur in the train set. In these cases
              # we can not find those values from the utility matrix.
              # That is why a check is necessary.
              # 1. both user and item in train
              # 2. only user in train
              # 3. only item in train
              # 4. none in train
              if user in self.users:
                     user_index = self.users.index(user)
                     if item in self.items:
                            item_index = self.items.index(item)
                            return self.utility_matrix[user_index, item_index]
                     else:
                            return self.user_means[user_index]
              elif item in self.items and user not in self.users:
                     item_index = self.items.index(item)
                     return self.item_means[item_index]
              else:
                     return np.mean(self.item_means)*0.6 + np.mean(self.user_means)*0.4
       
       def reco(self, user, repeat = False, k = 3):
              """
              Recommends the top k items to the user

              Parameters:
                     user : string or integer
                            The value that you used to label the user

                     repeat : boolean, default = False
                            Whether to allow the repeating of items that 
                            the user has already rated before

                     k : integer, default = 3
                            The number of items to recommend to the user
              
              Returns:
                     output : list
                            A list containing k + 1 elements
              """
              try:
                     user_index = self.users.index(user)
              except:
                     raise AttributeError('User does not exist')
              users_ratings = self.utility_matrix[user_index, :]
              sorted_array_index = np.argsort(users_ratings)[::-1]
              if repeat:
                     return [['Item', 'Rating']] + [[self.items[sorted_array_index[i]], users_ratings[sorted_array_index[i]]] for i in range(k)]
              else:
                     history_index = [item for item in self.user_history[user]]
                     result = [['Item', 'Rating']]
                     i = 0
                     while len(result) <= k + 1:
                            item_index = sorted_array_index[i]
                            if item_index not in history_index:
                                   result.append([self.items[item_index], users_ratings[item_index]])
                            i += 1
                     return result
