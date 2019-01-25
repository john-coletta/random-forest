# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:55:08 2019

@author: johnb

My attempt at creating my own random forest class
"""
# Import libraries
import numpy as np
import pandas as pd

class my_RandomForestClassifier():
    '''
    This object is a random forest binary classifier that takes the data directly
    as an arguement (so no fit method) and runs a random forest algorithm.
    Lot's of help from a towardsdatascience article
    '''
    
    def __init__(self, x, y, n_trees, n_features, sample_size, depth=5, min_leaf=5):
        # x is the predictor variables
        # y is the target
        # n_trees is the number of trees in the random forest
        # n_features is how many features to consider each tree
        # sample_size is the sample of data to build each tree on
        # depth is the max depth of the tree
        # min_leaf is the minimum number of observations for a leaf to split
        
        # Set the see
        np.random.seed(69)
        
        # Set the number of features, either sqrt, log2, or all features
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
            
        else:
            self.n_features = n_features
        
        
        
        
        