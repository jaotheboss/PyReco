# PyReco
Python recommendation system that does not rely on C-ython

## Motivation:
I started this project because I was looking for a recommendation system for Python. However, my search has led me to multiple systems that require the installation of Cython which does not sit well with my local computer. Therefore, I set out to code out a recommendation system without the need for Cython! 

## Description:
This module only requires modules that does not rely on C (in the sense that you don't need Cython). I've tried to emulate as much as the math as I could through the use of Numpy arrays. I have hence created 3 kinds of recommendation systems:
 
1. User-Based Collaborative Filtering (UBCF)
2. Item-Based Collaborative Filtering (IBCF)
3. SVD Filtering

### UBCF:
The gist of this method of recommendation is to find users that are considered 'similar' based on their rating patterns. Afterwhich, an average of scores will be taken from those similar users to represent the predicted score of the target user. For example, we want to find the predicted score of user A for Movie 1. Based on UBCF, user A has similar rating patterns with user's B to G, all who have watched Movie 1 already. Given that they have the same rating patterns, we assume that user A would rate Movie 1 similar to those of user's B to G. Hence, we take the ratings of user's B to G for Movie 1 and average them out and take that value to be the predicted rating value for user A. The number of similar users can be tweaked in the model. Whether to weigh the ratings of the similar users (higher weightage for users that have a more similar rating pattern, lower for those that don't) is also available for tweaking. 

In summary, products are recommended to the target user based on what similar users like.


### IBCF:
IBCF is similar to UBCF except for the fact that it takes the perspective of the item compared to the user. Instead of recommending products based on what similar users like, IBCF recommends products that are similar to products that the target user likes. The idea is that if a user rates 5/5 for a particular product, IBCF would look for products that have similar rating patterns. It is important to note that it is not about the proportion of ratings it gets but the actual pattern from each user. For example, if 5 users rated item A as [5, 5, 4, 3, 1], item B [5, 4, 5, 2, 2] and item C [1, 3, 5, 4, 5], although item A and C have similar ratings, their fixed pattern is not similar at all. Instead item A and B are the most similar because the users that rated item A rated the same way for item B. 

Basically, instead of answering 'Who else likes the same thing as this target user?', IBCF answers 'What is the demographic of users that like and dislike the item?', and based on that comparison, find another item that is similar that the target user hasn't tried yet. 

### SVD Filtering:
SVD works on the basis of a mathematical correlation between user and item based on the nature of the dataset, which means every thing. From the mathematical method of interpolating, the predictions of a user's rating for an item is calculated. This is the most layman way I can explain. However, if you would like to know more, you can read up on this ![page](https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm).

## Results:
[Movielens (100k)](https://grouplens.org/datasets/movielens/100k/)|RMSE|MAE|Time (Fit)
-|-|-|-
SVD|0.92759|0.71361|45s
kNN (UBCF/IBCF)|1.01870|0.78215|13s

Evaluation is done by splitting the 100k dataset into 90% training and 10% test. The systems are then fitted with the 90% training data and evaluated based on the 10% test data.

## How to use:

Data Preparation - Looking at the original form vs. the pivoted form

The original dataset usually looks something like this. 

![Original](https://github.com/jaotheboss/PyReco/blob/master/Original%20Data.png)

The data would need to be pivoted as shown below:
```
data = pd.read_csv('ratings.csv')
data.pivot_table(values = 'rating', index = 'userId', columns = 'movieId')
```

![Pivoted](https://github.com/jaotheboss/PyReco/blob/master/Pivoted%20Data.png)

Afterwhich, the execution is simple:

#### SVD Recommendation System
```
# for SVD recommendation system
from PyReco import SVDRecommender    # importing the class

recommender = SVDRecommender()       # creating an instant
recommender.fit(train)               # fitting the system with whatever training data

# for predicting the rating of an item from a user
recommender.predict('user', 'item')  

# for retrieving recommendations for a particular user
recommender.reco('user')
```

#### Collaborative Filtering Recommendation System
```
# for SVD recommendation system
from PyReco import CFRecommender          # importing the class

recommender = CFRecommender('ubcf or ibcf')       # creating an instant
recommender.fit(train)                    # fitting the system with whatever training data

# for predicting the rating of an item from a user
recommender.predict('user', 'item')  

# for retrieving recommendations for a particular user
recommender.reco('user')

# for finding the top most similar item or user from the input item or user
recommender.top_matches('user or item)
```
