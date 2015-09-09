#!/usr/bin/python

'''
This file tries to map correlation between columns in the data frame
Not used in the main code but for experimentation

TODO: Include Frequent Itemsets to find the col attributes that are similar 
to IsBadBuy in the training set 
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold

import preprocess

class correlation:

    def __init__(self):
        self.df = pd.read_csv('data/testing.csv', header=0)
        self.df = preprocess.preprocess(self.df)
        self.veh_year_purch()
        self.regressions(model='linear')

    def veh_year_purch(self):
        '''
        This function ascertains that Vehicle Year + Vehicle Age = Year of Vehicle Purchase
        '''
        self.df['Year_add_Age'] = self.df['VehYear'] + self.df['VehicleAge']
        self.df['PurchYear'] = df['PurchDate'].map(lambda x : x.split('/')[2])
        self.df['PurchYear_eq_VehAge'] = self.df['PurchYear'] == self.df['Year_add_Age']
        count = self.df[['PurchYear_eq_VehAge']].count()[0]
        if count == len(self.df.index):
            print "Purchase Year = Vehicle Year + Vehicle Age"
        else: 
            print "Purchase Year != Vehicle Year + Vehicle Age"

    def regressions(self, model='linear'):
        '''
        This function print the values ordered by correlation from logistic log_regressions
        '''
        
        if model == 'linear':
            reg_model = LinearRegression()
        elif model == 'log':
            reg_model = LogisticRegression(random_state=1)
        else:
            print "Model should be either 'linear' or 'log'"
            return
            
        variables = list(self.df.columns.values)[1:]
        scores_mean = []
                
        for var in variables: 
            scores = []
            scores = cross_validation.cross_val_score(reg_model, self.df[[var]], self.df['IsBadBuy'], cv=10)
            scores_mean.append((var, abs(scores.mean())))
        
        sorted_scores = sorted(scores_mean, key=lambda score: score[1], reverse=True)
        print [var[0] for var in sorted_scores]
        
    def frequent_itemsets(self):
        '''
        This function is to find correlations between desired variable and other variables
        To use for feature engineering
        '''
        pass
