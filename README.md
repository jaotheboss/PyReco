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
[pending explanation]


### IBCF:
[pending explanation]


### SVD Filtering:
[pending explanation]

## Results:
[Movielens (100k)](https://grouplens.org/datasets/movielens/100k/)|RMSE|MAE|Time (Fit)
-|-|-|-|-
SVD|0.92759|0.71361|45s
kNN (UBCF/IBCF)|1.01870|0.78215|13s

Evaluation is done by splitting the 100k dataset into 90% training and 10% test. The systems are then fitted with the 90% training data and evaluated based on the 10% test data.

## How to use:

Data Preparation - Looking at the original form vs. the pivoted form

[Original]()

[Pivoted]()
