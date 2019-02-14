# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:55:08 2019

@author: johnb

My attempt at creating my own random forest class
"""
# Import libraries
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
import pandas as pd
import math

class my_RandomForestRegressor():
    '''
    This object is a random forest regressor that takes the data directly
    as an arguement (so no fit method) and runs a random forest algorithm.
    Lot's of help from a towardsdatascience article
    '''
    
    def __init__(self, x, y, n_trees, n_features, sample_size, depth=10, min_leaf=5):
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
        self.x, self.y, self.sample_size, self.depth, self.min_leaf = x, y, int(np.floor(sample_size * len(y))), depth, min_leaf
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
        return DecisionTreeRegressor(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
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
    
    
def std_agg(cnt, s1, s2):
    '''
    This function calculates the standard deviation
    of two halves of our data (for used in determining
    the best split in our decision tree class)
    '''
    return math.sqrt((s2/cnt) - (s1/cnt)**2)


class DecisionTreeRegressor():
    '''
    This class is a basic decision tree that takes data and 
    produces and stores the splits
    '''
    def __init__(self, x, y, n_features, f_idxs, idxs, depth=10, min_leaf=5):
        '''
        Initialize the tree with x, y, the number of features, the index of the features, 
        the index of the x's, the depth and the min_leaf size
        '''
        # In case the indexes are not defined just take all the rows
        if idxs is None:
            idxs = np.arange(len(y))
            
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        # Here get the number of observations and the number of columns
        self.n, self.c = len(idxs), x.shape[1]
        # Get the value for the leaf (this should be changed to voting not averaging)
        self.val = np.mean(y[idxs])
        #print(self.val)
        # Score is how well a split divides the original set, initially set to infinity
        self.score = float('inf')
        # Perform the split (defined later)
        self.find_varsplit()
        
    def find_varsplit(self):
        # Find the split (recursive)
        for i in self.f_idxs:
            #print(i)
            # Find any better split
            self.find_better_split(i)
        # Check if we are in a leaf
        if self.is_leaf:
            return
        # Here we set which column to split on
        x = self.split_col
        # Here we set the observations that the left and right hand sides of the tree
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        # Set the features that the left and right trees have (bagging)
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        # Define the left and right hand sides as more trees, each time it finds the best split
        self.lhs = DecisionTreeRegressor(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth-1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTreeRegressor(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth-1,
                                min_leaf=self.min_leaf)
            
    def find_better_split(self, var_idx):
        #print(var_idx)
        # Get the values for the column and the target
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        #print(x)
        #print(y)
        # Get the indices to sort on the predictor column
        sort_idx = np.argsort(x)
        #print(sort_idx)
        # Sort both target and predictor
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]
        # Set the initial split (i.e. no split)
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0.0, 0.0
    
        for i in range(0, self.n-self.min_leaf-1):
            '''In this loop we loop through all values for a given leaf
            and split on that, calculated the weighted standard deviation for this
            split, and compare to the previous lowest value. If it is the lowest,
            it sets the split at this value.
            The first step is to sort the column so that we can iterate through the values
            to split the data each time. We calculate the standard deviation for each split and store the
            index associated with that split.
            '''
            # Pull the value from the sorted list
            xi, yi = sort_x[i], sort_y[i]
            # Increment the counts for the left and right nodes
            lhs_cnt += 1
            rhs_cnt -= 1
            # Increment the sum of values in the left and right nodes
            lhs_sum += yi
            rhs_sum -= yi
            # Increment the sum of squared values in the left and right nodes
            lhs_sum2 += yi**2
            rhs_sum2 -= yi**2
            # Make sure our min leaf criteria is not violated
            if i < self.min_leaf or xi == sort_x[i+1]:
                continue
            
            # Get the weighted standard deviation for this split as curr_score
            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            # If this score is better, update the split to xi
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi
                
    @property
    def split_name(self):
        # Get the name of the column we split on
        return self.x.columns[self.var_idx]
    
    @property
    def split_col(self):
        # Get the values for this column we have split on
        return self.x.values[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self):
        # Used to determin if a node is a leaf or not 
        # (either no split (so score is inf) or depth <= 0 so reached max depth)
        return self.score == float('inf') or self.depth <= 0
    
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        
        return t.predict_row(xi)

if __name__ == '__main__':
    
    diabetes = load_diabetes()
    data = np.c_[diabetes.data, diabetes.target]
    df = pd.DataFrame(data, columns=np.append(diabetes['feature_names'],'target'))
    var = df.drop('target', axis=1)
    target = df.target.values
    
    forest1 = my_RandomForestRegressor(var, target, n_trees=10, n_features='sqrt', sample_size=0.9, depth=10, min_leaf=5)  
    
    preds = forest1.predict(var.values)
    
    error = np.mean((preds - target)**2)
    print(np.sqrt(error))
        
diabetes = load_diabetes()
data = np.c_[diabetes.data, diabetes.target]
df = pd.DataFrame(data, columns=np.append(diabetes['feature_names'],'target'))
var = df.drop('target', axis=1)
target = df.target.values   
    
forest2 = my_RandomForestRegressor(var, target, n_trees=1000, n_features='sqrt',
                                   sample_size=1, depth=10, min_leaf=5)

preds2 = forest2.predict(var.values)
error2 = np.mean((preds2-target)**2)
print(np.sqrt(error2))

####################################
### The below code is for          #
### a random forest classification #
### task. ##########################
####################################

class my_RandomForestClassifier():
    '''
    This object is a random forest classifier that takes the data directly
    as an arguement (so no fit method) and runs a random forest algorithm.
    Lot's of help from a towardsdatascience article on creating a regressor
    '''
    
    def __init__(self, x, y, n_trees, n_features, sample_size, depth=10, min_leaf=5):
        # x is the predictor variables
        # y is the target
        # n_trees is the number of trees in the random forest
        # n_features is how many features to consider each tree
        # sample_size is the sample of data to build each tree on
        # depth is the max depth of the tree
        # min_leaf is the minimum number of observations for a leaf to split
        '''
        Everything should be very similar to the regression case
        besides the predictio being the majority voting (mode)
        instead of the mean
        '''
        
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
        self.x, self.y, self.sample_size, self.depth, self.min_leaf = x, y, int(np.floor(sample_size * len(y))), depth, min_leaf
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
        return DecisionTreeClassifier(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
                            idxs=np.array(range(self.sample_size)), depth=self.depth, min_leaf=self.min_leaf)
        
        
    def predict(self, x):
        '''
        This is the predict method that takes the mode (majority voting)
        of all the created trees
        '''
        return np.mode([t.predict(x) for t in self.trees], axis=0)
    
    
def entropy(node):
    '''
    This function calculates the entropy of a node for use
    later in determining information gain
    Here node is simply the array of target
    (so that the mean is the percent 1)
    '''
    p1 = np.mean(node)
    p2 = 1 - p1
    
    ent1 = (-1) * (p1 * np.log2(p1))
    ent2 = (-1) * (p2 * np.log2(p2))
    return (ent1 + ent2)


class DecisionTreeClassifier():
    '''
    This class is a basic decision tree that takes data and 
    produces and stores the splits (binary classifcation case)
    '''
    def __init__(self, x, y, n_features, f_idxs, idxs, depth=10, min_leaf=5):
        '''
        Initialize the tree with x, y, the number of features, the index of the features, 
        the index of the x's, the depth and the min_leaf size
        '''
        # In case the indexes are not defined just take all the rows
        if idxs is None:
            idxs = np.arange(len(y))
            
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        # Here get the number of observations and the number of columns
        self.n, self.c = len(idxs), x.shape[1]
        # Get the value for the leaf (by voting)
        self.val = np.mode(y[idxs])
        #print(self.val)
        # Score is how well a split divides the original set, initially set to the entropy
        # of the node
        self.score = entropy(y)
        # Perform the split (defined later)
        self.find_varsplit()
        
    def find_varsplit(self):
        # Find the split (recursive)
        for i in self.f_idxs:
            #print(i)
            # Find any better split
            self.find_better_split(i)
        # Check if we are in a leaf
        if self.is_leaf:
            return
        # Here we set which column to split on
        x = self.split_col
        # Here we set the observations that the left and right hand sides of the tree
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        # Set the features that the left and right trees have (bagging)
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        # Define the left and right hand sides as more trees, each time it finds the best split
        self.lhs = DecisionTreeClassifier(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth-1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTreeClassifier(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth-1,
                                min_leaf=self.min_leaf)
            
    def find_better_split(self, var_idx):
        #print(var_idx)
        # Get the values for the column and the target
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        #print(x)
        #print(y)
        # Get the indices to sort on the predictor column
        sort_idx = np.argsort(x)
        #print(sort_idx)
        # Sort both target and predictor
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]
        # Set the initial split (i.e. no split)        
        rhs_cnt, rhs_y = self.n, sort_y
        lhs_cnt, lhs_y = 0, []
    
        for i in range(1, self.n-self.min_leaf-1):
            '''In this loop we loop through all values for a given leaf
            and split on that, calculated the weighted standard deviation for this
            split, and compare to the previous lowest value. If it is the lowest,
            it sets the split at this value.
            The first step is to sort the column so that we can iterate through the values
            to split the data each time. We calculate the standard deviation for each split and store the
            index associated with that split.
            '''
            # Pull the value from the sorted list
            xi = sort_x[i]
            rhs_y = sort_y[i:]
            lhs_y = sort_y[:i]
            # Increment the counts for the left and right nodes
            lhs_cnt += 1
            rhs_cnt -= 1
            # Make sure our min leaf criteria is not violated
            if i < self.min_leaf or xi == sort_x[i+1]:
                continue
            
            # Get the weighted standard deviation for this split as curr_score
            lhs_ent = entropy(lhs_y)
            rhs_ent = entropy(rhs_y)
            curr_score = ((lhs_ent * lhs_cnt) + (rhs_ent * rhs_cnt)) / (lhs_cnt + rhs_cnt)
            # If this score is better, update the split to xi
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi
                
    @property
    def split_name(self):
        # Get the name of the column we split on
        return self.x.columns[self.var_idx]
    
    @property
    def split_col(self):
        # Get the values for this column we have split on
        return self.x.values[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self):
        # Used to determin if a node is a leaf or not 
        # (either no split (so score is inf) or depth <= 0 so reached max depth)
        return self.score == float('inf') or self.depth <= 0
    
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        
        return t.predict_row(xi)
        
        
        
        