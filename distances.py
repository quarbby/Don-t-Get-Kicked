#!/usr/bin/python

'''
This file implements three common distance measures 
1. Euclidean
2. Manhattan
3. Cosine 

Inputs: Vectors x (numpy array), Vector Y (numpy array)
Output: Distance between the two vectors (float)
'''

import numpy as np
import scipy

from scipy import sparse

def euclidean_distance(x, y):
	if scipy.sparse.issparse(x):
		return np.linalg.norm((x-y).toarray().ravel())
	else:
		return np.linalg.norm(x-y)

def manhattan_distance(x, y):
    if scipy.sparse.issparse(x):
        return np.sum(np.absolute((x-y).toarray().ravel()))
    else:
        return np.sum(np.absolute(x-y))	
		
def cosine_distance(x, y):
    if scipy.sparse.issparse(x):
        x = x.toarray().ravel()
        y = y.toarray().ravel()
    return 1.0 - np.dot(x, y)	