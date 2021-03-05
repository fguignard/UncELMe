# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:21:53 2017

@author: fguignar
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .base import _Activation_function

class ELM(BaseEstimator, RegressorMixin):
    
    def __init__(self, n_neurons=100, activation='logistic', 
                 weight_distr='uniform', weight_scl=1.0, random_state=None):
        '''
        Parameters
        ----------
        n_neurons : integer,
            Number of neurons. The default is 100.
        activation : string, optional
            Activation function ('logistic' or 'tanh'). The default is 'logistic'.
        weight_distr : string, optional
            Distribution of weights ('uniform' or 'gaussian'). 
            The default is 'uniform'. 
        weight_scl : float, optional
            Control the scale of the weight distribution.
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
    
    def _weight_draw(self):        
        '''
        Draw random input weights of the hidden layer.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
            
        '''    
    
        n_feat = self.X_.shape[1]
        
        np.random.seed(self.random_state)
        
        if self.weight_distr == 'uniform':
            drawing = np.random.uniform(-self.weight_scl, self.weight_scl, 
                                        (n_feat+1, self.n_neurons))
        elif self.weight_distr == 'gaussian':
            drawing = np.random.normal(0, self.weight_scl, 
                                       (n_feat+1, self.n_neurons))
        else :
            raise TypeError("Only 'uniform' and 'gaussian' are available for the 'weight_distr' argument")
        
        self.coef_hidden_ = drawing[:-1,:]   # random input weights, #neurons x #features, array
        self.intercept_hidden_ = drawing[-1,:] # random bias, #neurons, array
        
        return 
    
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
        
        Reference
        ---------
        G.-B. Huang, Q.-Y. Zhu, C.-K. Siew, 
        Extreme learning machine: theory and applications, 
        Neurocomputing 70 (1-3) (2006) 489–501.  
        
        '''
        
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        X = np.matrix(X)
        n_obs, n_feat = X.shape
        
        self._weight_draw()
                
        self.H_ = self._H_compute(X)
        
        self.H_pinv_ = np.linalg.pinv(self.H_, rcond = np.sqrt(np.finfo(float).eps)) 
            
        y = np.matrix(y).T
        self.coef_output_ = np.array(self.H_pinv_ * y).squeeze() # neurons x #resp, array

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
        Output_weight = np.matrix(self.coef_output_).T
        
        y_predict = np.squeeze(np.asarray(H_predict * Output_weight))
                
        return y_predict
    
    
class ELMRidge(ELM, BaseEstimator, RegressorMixin):
    
    def __init__(self, n_neurons=100, alpha=1.0, activation='logistic', 
                 weight_distr='uniform', weight_scl=1.0, random_state=None):
        '''
        Parameters
        ----------
        n_neurons : integer,
            Number of neurons. The default is 100.
        alpha : float, optional
            Regularization strength, aka Tikhonov factor. The default is 1.0.
        activation : string, optional
            Activation function ('logistic' or 'tanh'). The default is 'logistic'.
        weight_distr : string, optional
            Distribution of weights ('uniform' or 'gaussian'). 
            The default is 'uniform'. 
        weight_scl : float, optional
            Control the scale of the weight distribution.
            If weights are uniforms, they are drawn from [-weight_scl, weight_scl].
            If weights are Gaussians, they are centred with standard deviation
            equal to weight_scl. The default is 1.0.
        random_state : integer, optional
            Random seed for reproductible results. The default is None.

        Returns
        -------
        None.

        '''
        
        ELM.__init__(self, n_neurons, activation, weight_distr, weight_scl, random_state)
        self.alpha = alpha                  
    
    def fit(self, X, y):
        '''
        Training for regularized ELM.
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
        
        Reference
        ---------
        W. Deng, Q. Zheng, L. Chen,  
        Regularized extreme learning machine, 
        IEEE symposium on computational intelligence and data mining, 2009, pp. 389–395.
        
        '''
        
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        X = np.matrix(X)
        n_obs, n_feat = X.shape
        
        self._weight_draw()
                
        self.H_ = self._H_compute(X)   

        H_Tikh = self.H_.T * self.H_ + self.alpha * np.identity(self.n_neurons)
        self.H_alpha_ = np.linalg.pinv(H_Tikh) * self.H_.T
            
        y = np.matrix(y).T
        self.coef_output_ = np.array(self.H_alpha_ * y).squeeze() # neurons x #resp, array

        return self
    

class ELMRidgeCV(ELM, BaseEstimator, RegressorMixin):
    
    def __init__(self, n_neurons=100, alphas=np.array([0.1, 1.0, 10]), activation='logistic', 
                 weight_distr='uniform', weight_scl=1.0, random_state=None):
        '''
        Parameters
        ----------
        n_neurons : integer,
            Number of neurons. The default is 100.
        alphas : ndarray, optional
            Array of alpha's values to try. Regularization strength, 
            aka Tikhonov factor. The default is np.array([0.1, 1.0, 10]).
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
        
        ELM.__init__(self, n_neurons, activation, weight_distr, weight_scl, random_state)
        self.alphas = alphas             
    
    def fit(self, X, y):
        '''
        Training for ridge ELM, with Generalized Cross Validation.
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
        
        References
        ----------
        W. Deng, Q. Zheng, L. Chen,  
        Regularized extreme learning machine, 
        IEEE symposium on computational intelligence and data mining, 2009, pp. 389–395.
        
        G. H. Golub, M. Heath, G. Wahba, 
        Generalized cross-validation as a method for choosing a good ridge parameter,
        Technometrics 21 (2) (1979) 215–223.
        
        '''
        
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        X = np.matrix(X)
        n_obs, n_feat = X.shape
        
        self._weight_draw()
                
        self.H_ = self._H_compute(X)
        
        eigenHTH = np.square(np.linalg.svd(self.H_ , full_matrices= False, compute_uv=False, hermitian=False))
        eigenHTH = eigenHTH.reshape(eigenHTH.shape[0], 1)
        trace = (eigenHTH/(eigenHTH + self.alphas)).sum(axis=0)      
        
        HTH = np.array(self.H_.T*self.H_)
        H_Tikh = HTH + self.alphas.reshape(self.alphas.shape[0], 1, 1) * np.identity(self.n_neurons)
        H_Tikh_inv = np.linalg.pinv(H_Tikh)
        
        y_hat = np.einsum('li, aij, kj, k -> al', self.H_, H_Tikh_inv, 
                          self.H_, y, optimize = 'greedy')
        self.GCV = np.linalg.norm(y-y_hat, axis = 1)
        self.GCV = n_obs * self.GCV /np.square(n_obs-trace)
        self.alpha_opt = self.alphas[np.argmin(self.GCV)]
        self.H_alpha_ = H_Tikh_inv[np.argmin(self.GCV)] * self.H_.T
                        
        y = np.matrix(y).T
        self.coef_output_ = np.array(self.H_alpha_ * y).squeeze() # neurons x #resp, array

        return self


