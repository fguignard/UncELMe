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
    M : numpy array
        Input matrix.
    func : string, optional
        Activation function (logistic or tanh). The default is 'logistic'.

    Returns
    -------
    M : numpy array
        Transformed input matrix.

    '''
    
    if func == 'logistic' :
        M = 1/(1+ np.exp(-M))
        
    elif func == 'tanh' :
        M = np.tanh(M)
    
    else :
        raise TypeError("Only 'logistic' or 'tanh' are available for the activation function")
        
    return M