# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:21:53 2017

@author: fguignar
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .base import _Activation_function

class ELMRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_neurons=100, alpha=None, activation='logistic', 
                 weight_distr='uniform', weight_scl=1.0, random_state=None):
        '''
        Parameters
        ----------
        n_neurons : integer,
            Number of neurons. The default is 100.
        alpha : float, optional
            Regularization strength, aka Tikhonov factor; must be a positive float. 
            The default is None, which means that no regularization  is performed.
        activation : string, optional
            Activation function ('logistic' or 'tanh'). The default is 'logistic'.
        weight_distr : string, optional
            Distribution of weights ('uniform' or 'gaussian'). 
            The default is 'uniform'. 
        weight_scl : float, optional
            Controle the scale of the weight distribution.
            If weights are uniforms, they are drawn from [-weight_scl, weight_scl].
            If weights are Gaussians, they are centred with standard deviation
            equal to weight_scl. The default is 1.0.
        random_state : integer, optional
            Random seed for reproductible results. The default is None.

        Returns
        -------
        None.

        '''
        
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.activation = activation
        self.weight_distr = weight_distr
        self.weight_scl = weight_scl
        self.random_state = random_state

    def _H_compute(self, X):        
        '''
        Parameters
        ----------
        X : Numpy array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        H : Numpy matrix of shape (n_samples, n_neurons)
            Data projected in the random feature space.
            
        '''    
    
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        n_obs = X.shape[0]
        X = np.matrix(X)
        
        Input_weight = np.matrix(self.coef_hidden_)
        Bias = np.matrix(self.intercept_hidden_)
        Bias_rep = Bias.repeat(n_obs, axis = 0)
        
        H = _Activation_function(X * Input_weight + Bias_rep,
                                  func = self.activation)
        
        return H
    
    def fit(self, X, y):
        '''
        Training for ELM.
        Initialize random hidden weights and compute output weights.
        
        Parameters
        ----------
        X : Numpy array of shape (n_sample_train, n_features)
            Training data.
        y : Numpy array of shape (n_sample_train)
            Target values.

        Returns
        -------
        Self.
        
        '''
        
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        X = np.matrix(X)
        y = np.matrix(y).transpose()
        n_obs, n_feat = X.shape
        
        # random drawing, seed
        np.random.seed(self.random_state)
        if self.weight_distr == 'uniform':
            Drawing = np.random.uniform(-self.weight_scl, self.weight_scl, (n_feat+1, self.n_neurons))
        elif self.weight_distr == 'gaussian':
            Drawing = np.random.normal(0, self.weight_scl, (n_feat+1, self.n_neurons))
        
        self.coef_hidden_ = Drawing[:-1,:]   # random input weights, #neurons x #features, array
        self.intercept_hidden_ = Drawing[-1,:] # random bias, #neurons, array
                
        self.H_ = self._H_compute(X)
        
        if self.alpha == None :  # this correspond to alpha = 1, and H_alpha_ is the Moore-Penrose inverse
            self.H_alpha_ = np.linalg.pinv(self.H_, rcond = np.sqrt(np.finfo(float).eps)) 
        else :
            H_Tikh = self.H_.transpose() * self.H_ + self.alpha * np.identity(self.n_neurons)
            self.H_alpha_ = np.linalg.pinv(H_Tikh) * self.H_.transpose()

        self.coef_output_ = np.array(self.H_alpha_ * y).squeeze() # neurons x #resp, array

        return self

    def predict(self, X_predict):
        '''
        Parameters
        ----------
        X_predict : Numpy array of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        y_predict : numpy array of shape (n_samples)
            Predicted output

        ''' 

        H_predict = self._H_compute(X_predict)
        Output_weight = np.matrix(self.coef_output_).transpose()
        
        y_predict = np.squeeze(np.asarray(H_predict * Output_weight))
                
        return y_predict
    