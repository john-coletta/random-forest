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
        # Print the features out    
        print(self.n_features, 'sha: ', x.shape[1])
        # Define the other variables from the input
        self.x, self.y, self.sample_size, self.depth, self.min_leaf = x, y, sample_size, depth, min_leaf
        # Build the list of trees with the create_tree function
        self.trees = [self.create_tree() for i in range(n_trees)]
        
    def create_tree(self):
        '''
        This is the function that creates the decision trees for the random forest
        '''
        # Get the indexes for the sample the tree will be built on
        idxs = np.random.permutation(len(self.y))[:self.sample_size]
        # Get the features for this tree
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        # This returns the tree
        return DecisionTree(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
                            idxs=np.array(range(self.sample_size)), depth=self.depth, min_leaf=self.min_leaf)
        '''
        ^This class will be defined below
        '''
        
    def predict(self, x):
        '''
        This is the predict method that takes the mean of all the created trees
        NEED TO MAKE INTO A CLASSIFIER
        '''
        return np.mean([t.predict(x) for t in self.trees], axis=0)
    
    
        
        
        
        
        
        