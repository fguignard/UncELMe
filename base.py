#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:19:23 2017

@author: fguignar1
"""

import numpy as np

def _Activation_function(M, func = 'logistic') :
    '''
    Parameters
    ----------
    M : numpy matrix
        Input matrix.
    func : string, optional
        Activation function (logistic or tanh). The default is 'logistic'.

    Returns
    -------
    M : numpy matrix
        Transformed input matrix.

    '''
    
    if func == 'logistic' :
        M = 1/(1+ np.exp(-M))
        
    elif func == 'tanh' :
        M = np.tanh(M)
    
    else :
        raise TypeError("Only 'logistic' or 'tanh' are available for the activation function")
        
    return M

def Timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total elapsed time : " + 
          "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return