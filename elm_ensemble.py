# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:42:29 2017

@author: fguignar
"""
import numpy as np
from .elm_estimator import ELMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ELMEnsemble(BaseEstimator, RegressorMixin):

    def __init__(self, n_estimators=20, n_neurons=100, alpha=None,
                 activation='logistic', weight_distr='uniform', weight_scl=1.0,
                 n_jobs=1, random_state=None):
        '''
        Parameters
        ----------
        n_estimators : integer,
            Number of ELM models in the ensemble. The default is 20.
        n_neurons : integer,
            Number of neurons of each model. The default is 100.
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
        n_jobs : integer,
            Number of processors to use for the computation.
        random_state : integer, optional
            Random seed for reproductible results. The default is None.

        Returns
        -------
        None.

        '''

        self.n_estimators = n_estimators
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.activation = activation
        self.weight_distr = weight_distr
        self.weight_scl = weight_scl
        self.n_jobs = n_jobs                        ## a paraleliser
        self.random_state = random_state


    def fit(self, X, y):
        '''
        Training for ELM ensemble.
        Initialize random hidden weights and compute output weights
        for all ELM models.

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

        self.check = X.shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        # random_state variable treatment
        if self.random_state == None :
            random_states = [None] * self.n_estimators
            random_states = np.array(random_states)
        else :
            np.random.seed(self.random_state)
            random_states = np.random.randint(int(1e8), size = self.n_estimators)

        # multiple fitting
        n_obs, n_feat = X.shape
        self.estimators_ = [0] * self.n_estimators
        for i in range(self.n_estimators):
            elm = ELMRegressor(n_neurons = self.n_neurons,
                               alpha = self.alpha,
                               activation = self.activation,
                               weight_distr = self.weight_distr,
                               weight_scl = self.weight_scl,
                               random_state = random_states[i])
            elm.fit(X,y)
            self.estimators_[i] = elm

        return self

    def predict(self, X_predict):
        '''
        Averaged prediction of ELM ensemble.

        Parameters
        ----------
        X_predict : Numpy array of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        y_predict : numpy array of shape (n_samples)
            Predicted output

        '''

        check_is_fitted(self, ['X_', 'y_'])
        X_predict = check_array(X_predict)

        self.X_predict_ = X_predict

        y_predict = np.zeros((X_predict.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
             y_predict[:,i] = self.estimators_[i].predict(X_predict)

        self.y_var_ = y_predict.var(axis = 1, ddof = 1)
        y_predict_avg = y_predict.mean(axis = 1)

        return y_predict_avg

    def model_variance(self, heterosk=False, estimator='naive'):
        '''
        Compute a variance estimation of the model at last predicted points.

        Parameters
        ----------
        heterosk : boolean, optional
            Compute the variance estimate assuming heteroskedasticity (non-constant
            noise variance). Default is False.
        estimator : string, optional
            Estimator used for the variance estimation. Default is 'naive'.

        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy array of shape (n_sample_train)
            Noise estimate at the training points.

        '''

        n_obs, n_feat = self.X_.shape
        X_predict = self.X_predict_
        n_predict = X_predict.shape[0]

        # Homoskedasic estimates
        if heterosk == False :

            # Naive estimate
            if estimator == 'naive' :

                # Collect H_alpha, H_predict and residuals in each model
                H_alphas = np.zeros((self.n_neurons, n_obs, self.n_estimators))
                H_predict = np.zeros((n_predict, self.n_neurons, self.n_estimators))
                sq_residuals = np.zeros((n_obs, self.n_estimators))

                for i in range(self.n_estimators):
                    elm = self.estimators_[i]
                    H_alphas[:, :, i] = elm.H_alpha_
                    H_predict[:, :, i] = elm._H_compute(X_predict)
                    sq_residuals[:, i] = np.square(elm.predict(self.X_) - self.y_)

                # Compute noise estimation
                ddof = n_obs - self.n_neurons
                noise = np.einsum('im->', sq_residuals)
                noise = noise / (ddof * self.n_estimators)

                #Compute model variance induced by noise
                var1 = np.einsum('nim, ijm, kjl, nkl -> n', H_predict, H_alphas, H_alphas, H_predict, optimize = 'greedy')
                var1 = var1 / (self.n_estimators**2)

                # Compute model variance induced by the random weights
                var2 = self.y_var_/self.n_estimators

                var_predict = noise * var1 + var2
                noise_estimate = noise * np.ones(n_obs)

            #  Estimate with reduced bias
            elif estimator == 'bias_reduced' :
                None

            else :
                # raised error
                None

        # Heteroskedasic estimates
        elif heterosk == True :

            # Naive estimate
            if estimator == 'naive' :
                # Collect H_, H_alpha, H_predict and residuals in each model
                H_alphas = np.zeros((self.n_neurons, n_obs, self.n_estimators))
                H_predict = np.zeros((n_predict, self.n_neurons, self.n_estimators))
                sq_residuals = np.zeros((n_obs, self.n_estimators))
                Hs = np.zeros((n_obs, self.n_neurons, self.n_estimators))  #heterosk
                for i in range(self.n_estimators):
                    elm = self.estimators_[i]
                    H_alphas[:, :, i] = elm.H_alpha_
                    H_predict[:, :, i] = elm._H_compute(X_predict)
                    sq_residuals[:, i] = np.square(elm.predict(self.X_) - self.y_)
                    Hs[:, :, i] = elm.H_                    #heterosk

                # Compute noise estimation for each model
                diag_projs = np.einsum('ijm, jim -> im', Hs, H_alphas)
                noise = sq_residuals/np.square(1-diag_projs)   #optimize the square with the above for loop

                #Compute model variance induced by noise
                var1 = np.einsum('nim, ijm, jm, kjl, nkl -> n', H_predict, H_alphas, noise, H_alphas, H_predict, optimize = 'greedy')
                var1 = var1 / (self.n_estimators**2)

                # Compute model variance induced by the random weights
                var2 = self.y_var_/self.n_estimators

                var_predict = var1 + var2
                noise_estimate = noise.mean(axis = 1)

            #  Estimate with reduced bias
            elif estimator == 'bias_reduced' :
                None

            else :
                # raised error
                None

        else :
            # raised error
            None

        return var_predict, noise_estimate
