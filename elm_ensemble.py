# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:42:29 2017

@author: fguignar
"""
import numpy as np
from .elm_estimator import ELM, ELMRidge, ELMRidgeCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ELMEnsemble(BaseEstimator, RegressorMixin):
    
    def __init__(self, n_estimators=20, n_neurons=100, 
                 activation='logistic', weight_distr='uniform', weight_scl=1.0, 
                 n_jobs=1, random_state=None):
        '''
        Parameters
        ----------
        n_estimators : integer,
            Number of ELM models in the ensemble. The default is 20.
        n_neurons : integer,
            Number of neurons of each model. The default is 100.
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
        print("ELMEnsemble __init__")
        self.n_estimators = n_estimators
        self.n_neurons = n_neurons
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
        print("ELMEnsemble fit")                
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
            elm = ELM(n_neurons = self.n_neurons,
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
        print("ELMEnsemble predict")                         
        check_is_fitted(self, ['X_', 'y_'])
        X_predict = check_array(X_predict)
        
        self.X_predict_ = X_predict
        
        y_predict = np.zeros((X_predict.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
             y_predict[:,i] = self.estimators_[i].predict(X_predict)
        
        self.y_var_ = y_predict.var(axis = 1, ddof = 1)   
        y_predict_avg = y_predict.mean(axis = 1)    
        
        return y_predict_avg
  
    def _collect(self):
        '''
        Collect H_pinv_, H_predict and residuals in each model

        Returns
        -------
        H_pinvs : numpy array of shape (n_neurons, n_samples, n_estimators)
            Pseudo-inverses of hidden matrices.
        H_predict : numpy array of shape (n_predicted_points, n_neurons, n_estimators)
            Hidden matrices for all the predicted points.
        residuals : numpy array of shape (n_samples, n_estimators)
            Training residuals for all models.
        '''
        print("ELMEnsemble _collect") 
        n_obs = self.X_.shape[0]
        n_predict = self.X_predict_.shape[0]
        N = self.n_neurons
        M = self.n_estimators
        
        H_pinvs = np.zeros((N, n_obs, M))
        H_predict = np.zeros((n_predict, N, M))
        residuals = np.zeros((n_obs, M))
        
        for i in range(M):   
            elm = self.estimators_[i]
            H_pinvs[:, :, i] = elm.H_pinv_
            H_predict[:, :, i] = elm._H_compute(self.X_predict_)
            residuals[:, i] = elm.predict(self.X_) - self.y_
            
        return H_pinvs, H_predict, residuals
    
    def _AvgRSS(self, residuals):
        '''
        Parameters
        ----------
        residuals : Numpy array of shape (n_samples, n_estimators)
            Training residuals for all ELMs. 

        Returns
        -------
        ARSS : Numpy matrix of shape (n_predict, n_samples)
            Residual sum of squares averaged over all ELMs.
            
        '''
        print("ELMEnsemble _AvgRSS") 
        sq_residuals = np.square(residuals)
        RSS = sq_residuals.sum(axis=0)
        ARSS = RSS.mean()
        
        return ARSS    
    
    def _muTmu_estim(self, H_pinvs, H_predict):
        '''
        Parameters
        ----------
        H_pinvs : Numpy array of shape (n_neurons, n_samples, n_estimators)
            Pseudoinverse matrices for all ELMs. (or H_alphas in the regularized case)
        H_predict : Numpy array of shape (n_predict, n_neurons, n_estimators)
            Hidden vectors at all predicted points for all ELMs.

        Returns
        -------
        muTmu : Numpy matrix of shape (n_predict)
            Unbiased estimate of the squared norm of mu
            
        References
        ----------
        Equation (10)
            
        '''    
        print("ELMEnsemble _muTmu_estim") 
        N = self.n_estimators
        
        z = np.einsum('pnm, nsm -> psm', H_predict, H_pinvs, optimize = 'greedy')
        mu = z.mean(axis=2)
    
        muTmu = np.einsum('ps, ps -> p ', mu, mu)
        muTmu *= N/(N-1)
        quad_terms = np.einsum('psm, psm -> p ', z, z)
        muTmu += - quad_terms/(N*(N-1))
        
        return muTmu
    
    def homoskedastic_variance(self):
        '''
        Compute an homoskedastic variance estimation of the model at last predicted points.
            
        Parameters
        ----------
        None.

        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy.float
            Noise estimate.
            
        Reference
        ---------
        F. Guignard, F. Amato and M. Kanevski (in prep). On Extreme Learning 
        Machine Model Variance.        
            
        ''' 
        print("ELMEnsemble homoskedastic_variance")                       
        n_obs = self.X_.shape[0]
        H_pinvs, H_predict, residuals = self._collect()
        
        # Compute noise estimation
        ddof = n_obs - self.n_neurons
        ARSS = self._AvgRSS(residuals)
        noise = ARSS / ddof
        
        #Compute model variance induced by noise
        muTmu = self._muTmu_estim(H_pinvs, H_predict)
        var1 = noise * muTmu
        
        # Compute model variance induced by the random weights
        var2 = self.y_var_/self.n_estimators

        var_predict = var1 + var2 
            
        return var_predict, noise 
  
    def _collect_Hs(self):
        '''
        Collect Hs in each model. Complete the _collect() function for the
        heteroskedastic case.

        Returns
        -------
        Hs : numpy array of shape (n_neurons, n_samples, n_estimators)
             Hidden matrices.

        '''
        print("ELMEnsemble _collect_Hs")  
        n_obs = self.X_.shape[0]
        N = self.n_neurons
        M = self.n_estimators
        
        Hs = np.zeros((n_obs, N, M))  
        for i in range(M):   
            elm = self.estimators_[i]
            Hs[:, :, i] = elm.H_  
            
        return Hs
    
    def _Sigma_estim(self, corr_res):
        '''
        Parameters
        ----------
        corr_res : Numpy array of shape (n_samples, n_estimators)
            Training corrected residuals for all ELMs. 

        Returns
        -------
        Sigma : Numpy matrix of shape (n_samples, n_samples, n_estimators)
            Estimates of noise covariance matrices for all ELMs,
            based on the ordinary Jackknife estimate (HC3) of the 
            output weights covariance matrix. 
            
        ''' 
        print("ELMEnsemble _Sigma_estim")  
        n_obs = corr_res.shape[0]
        rrT = np.einsum('im, jm -> ijm', corr_res, corr_res, optimize = 'greedy')
        Sigma = np.zeros(rrT.shape)
        for m in range(Sigma.shape[2]):
            Sigma[:,:,m] = np.diag(np.square(corr_res[:, m]))
        Sigma += -rrT/n_obs
        Sigma *= ((n_obs-1)/n_obs)
        
        return Sigma
    
    def _S1_estim(self, z, Sigma):
        '''
        Parameters
        ----------
        z : Numpy matrix of shape (n_predict, n_samples, n_estimators)
            "z" vectors at all predicted points for all ELMs.
        Sigma : Numpy matrix of shape (n_samples, n_samples, n_estimators)
            Estimates of noise covariance matrices for all ELMs.

        Returns
        -------
        S1 : Numpy matrix of shape (n_predict, n_samples)
            S1 estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _S1_estim")  
        S1 = np.einsum('pim, ijm, pjm -> p', z, Sigma, z, optimize = 'greedy')
        S1 = S1 /self.n_estimators
        
        return S1 
    
    def _nu_estim(self, z, Sigma):
        '''
        Parameters
        ----------
        z : Numpy matrix of shape (n_predict, n_samples, n_estimators)
            "z" vectors at all predicted points for all ELMs.
        Sigma : Numpy matrix of shape (n_samples, n_samples, n_estimators)
            Estimates of noise covariance matrices for all ELMs.
            
        Returns
        -------
        nu : Numpy matrix of shape (n_predict, n_samples)
            nu estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _nu_estim") 
        nu = np.einsum('pim, ijm -> pj', z, Sigma, optimize = 'greedy')
        nu = nu / self.n_estimators
        return nu
    
    def _S2_estim(self, mu, nu, S1):
        '''
        Parameters
        ----------
        mu : Numpy matrix of shape (n_predict, n_samples)
            Average of z across all ELMs.
        nu : Numpy matrix of shape (n_predict, n_samples)
            nu estimate, see section 3.3.
        S1 : Numpy matrix of shape (n_predict, n_samples)
            S1 estimate.

        Returns
        -------
        S2 : Numpy matrix of shape (n_predict, n_samples)
            S2 estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _S2_estim") 
        nuTmu = np.einsum('ps, ps -> p', mu, nu, optimize = 'greedy')
        S2 = self.n_estimators * nuTmu - S1
        S2 = S2 / (self.n_estimators - 1)
        
        return S2
    
    def _V_estim(self, z, U):
        '''
        Parameters
        ----------
        z : Numpy matrix of shape (n_predict, n_samples, n_estimators)
            "z" vectors at all predicted points for all ELMs.
        U : Numpy matrix of shape (n_samples, n_samples)
            Average of the noise covariance matrices estimates.

        Returns
        -------
        V : Numpy matrix of shape (n_predict)
            Averaged quadratic form in z, see section 3.3.
            
        '''    
        print("ELMEnsemble _V_estim") 
        V = np.einsum('pim, ij, pjm -> p', z, U, z, optimize = 'greedy')
        V = V /self.n_estimators
        
        return V 

    def _S3_estim(self, mu, U, V, S2):
        '''
        Parameters
        ----------
        mu : Numpy matrix of shape (n_predict, n_samples)
            Average of z across all ELMs.
        U : Numpy matrix of shape (n_samples, n_samples)
            Average of the noise covariance matrices estimates.
        V : Numpy matrix of shape (n_predict)
            Averaged quadratic form in z,
        S2 : Numpy matrix of shape (n_predict, n_samples)
            S2 estimate.

        Returns
        -------
        S3 : Numpy matrix of shape (n_predict, n_samples)
            S3 estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _S3_estim") 
        muTUmu = np.einsum('pi, ij, pj -> p', mu, U, mu, optimize = 'greedy')
        S3 = self.n_estimators**2 * muTUmu - self.n_estimators*V - 2*(self.n_estimators-1)*S2
        S3 = S3 / ((self.n_estimators - 1) * (self.n_estimators - 2))
        
        return S3
    
    def _S1_approx(self, z, Omega):
        '''
        Parameters
        ----------
        z : Numpy matrix of shape (n_predict, n_samples, n_estimators)
            "z" vectors at all predicted points for all ELMs.
        Omega : Numpy matrix of shape (n_samples, n_estimators)
            Estimates of noise covariance matrix diagonals for all ELMs.

        Returns
        -------
        S1_app : Numpy matrix of shape (n_predict, n_samples)
            Approximation of S1 estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _S1_approx") 
        S1_app = np.einsum('psm, sm, psm -> p', z, Omega, z, optimize = 'greedy')
        S1_app = S1_app/self.n_estimators
        
        return S1_app
              
    def _nu_approx(self, z, Omega):
        '''
        Parameters
        ----------
        z : Numpy matrix of shape (n_predict, n_samples, n_estimators)
            "z" vectors at all predicted points for all ELMs.
        Omega : Numpy matrix of shape (n_samples, n_estimators)
            Estimates of noise covariance matrix diagonals for all ELMs.
            
        Returns
        -------
        nu_app : Numpy matrix of shape (n_predict, n_samples)
            approximation of nu estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _nu_approx") 
        nu_app = np.einsum('psm, sm -> ps', z, Omega, optimize = 'greedy')
        nu_app = nu_app / self.n_estimators
        return nu_app

    def _V_approx(self, z, U_app):
        '''
        Parameters
        ----------
        z : Numpy matrix of shape (n_predict, n_samples, n_estimators)
            "z" vectors at all predicted points for all ELMs.
        U : Numpy matrix of shape (n_samples)
            Average of the noise covariance matrices diagonal estimates.

        Returns
        -------
        V_app : Numpy matrix of shape (n_predict)
            Averaged quadratic form in z, see section 3.3.
            
        '''    
        print("ELMEnsemble _V_approx") 
        V_app = np.einsum('psm, s, psm -> p', z, U_app, z, optimize = 'greedy')
        V_app = V_app /self.n_estimators
        
        return V_app 

    def _S3_approx(self, mu, U_app, V_app, S2_app):
        '''
        Parameters
        ----------
        mu : Numpy matrix of shape (n_predict, n_samples)
            Average of z across all ELMs.
        U_app : Numpy matrix of shape (n_samples)
            Average of the noise covariance matrices diagonal estimates.
        V_app : Numpy matrix of shape (n_predict)
            Averaged quadratic form in z,
        S2_app : Numpy matrix of shape (n_predict, n_samples)
            Approximate S2 estimate.

        Returns
        -------
        S3 : Numpy matrix of shape (n_predict, n_samples)
            Approximation of S3 estimate, see section 3.3.
            
        '''    
        print("ELMEnsemble _S3_approx") 
        M = self.n_estimators
        muTUmu = np.einsum('ps, s, ps -> p', mu, U_app, mu, optimize = 'greedy')
        S3_app = M**2 * muTUmu - M*V_app - 2*(M-1)*S2_app
        S3_app = S3_app / ((M-1) * (M-2))
        
        return S3_app    

       
    def heteroskedastic_variance(self, estimate='old', approx=False, corrected_residuals=False):
        '''
        Compute a variance estimation of the model at last predicted points.
            
        Parameters
        ----------
        estimate : string, optional
            Compute the variance estimate assuming heteroskedasticity (non-constant
            noise variance). Default is False.                          ?!?!??!?!!?
        approx : bool, optional
            Speed up computation at the expense of a sligntly approximate variance 
            estimation. Default is 'False'.
        corrected_residuals : bool, optional
            If True, the corrected residuals used to compute HC3 are returned as
            noise estimates at training points. If False, the raw residuals are 
            returned. Default is 'False'.
            
        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy array of shape (n_sample_train) 
            Noise estimate at the training points.
            
        References
        ----------
        F. Guignard, F. Amato and M. Kanevski (in prep). On Extreme Learning 
        Machine Model Variance.
        
        J.G. MacKinnon, H. White (1985). Some heteroskedasticity-consistent 
        covariance matrix estimators with improved finite sample properties, 
        Journal of econometrics, 29, 3, 305--325. 
                    
        ''' 
        print("ELMEnsemble heteroskedastic_variance") 
        n_obs, n_feat = self.X_.shape
        X_predict = self.X_predict_
        n_predict = X_predict.shape[0]
                                    
        # Heteroskedasic estimates
        if estimate == 'old' :
            
            if approx == True :
                # Collect H_, H_pinv_, H_predict and residuals in each model
                None
                
            elif approx == False :
                H_pinvs = np.zeros((self.n_neurons, n_obs, self.n_estimators))
                H_predict = np.zeros((n_predict, self.n_neurons, self.n_estimators))
                sq_residuals = np.zeros((n_obs, self.n_estimators))
                Hs = np.zeros((n_obs, self.n_neurons, self.n_estimators))  #heterosk
                for i in range(self.n_estimators):   
                    elm = self.estimators_[i]
                    H_pinvs[:, :, i] = elm.H_pinv_
                    H_predict[:, :, i] = elm._H_compute(X_predict)
                    sq_residuals[:, i] = np.square(elm.predict(self.X_) - self.y_)
                    Hs[:, :, i] = elm.H_                    #heterosk
                    
                # Compute noise estimation for each model
                diag_projs = np.einsum('ijm, jim -> im', Hs, H_pinvs)
                noise = sq_residuals/np.square(1-diag_projs)   #optimize the square with the above for loop
                
                #Compute model variance induced by noise
                var1 = np.einsum('nim, ijm, jm, kjl, nkl -> n', H_predict, H_pinvs,
                                 noise, H_pinvs, H_predict, optimize = 'greedy')
                var1 = var1 / (self.n_estimators**2)
                
                # Compute model variance induced by the random weights
                var2 = self.y_var_/self.n_estimators

                var_predict = var1 + var2
                noise_estimate = noise.mean(axis = 1)                 

            else :
                # raised error
                None

        elif estimate == 'S1' or estimate == 'S2' or estimate == 'S3' :

            H_pinvs, H_predict, residuals = self._collect()
            Hs = self._collect_Hs()
            
            z = np.einsum('pnm, nsm -> psm', H_predict, H_pinvs, optimize = 'greedy')
            Pdiag = np.einsum('nsm, snm -> nm', Hs, H_pinvs, optimize = 'greedy')
            corr_res = residuals/(1-Pdiag)
            
            # Noise estimate
            if corrected_residuals == False:
                noise_estimate = residuals.mean(axis=1)
                
            elif corrected_residuals == True:
                noise_estimate = corr_res.mean(axis=1)
                
            else :
                raise TypeError("Only booleans are allowed for the 'corrected_residuals' argument")

            # HC3 Jacknife estimation
            if approx == False :
                Sigma = self._Sigma_estim(corr_res)
                var1 = self._S1_estim(z, Sigma)                   #S1 estimate
                if estimate != 'S1':
                    mu = z.mean(axis=2)
                    nu = self._nu_estim(z, Sigma)
                    var1 = self._S2_estim(mu, nu, var1)           #S2 estimate
                    if estimate != 'S2' :
                        U = Sigma.mean(axis=2)
                        V = self._V_estim(z, U)
                        var1 = self._S3_estim(mu, U, V, var1)     #S3 estimate
                         
            elif approx == True :                                 #Simplified Jackknife
                Omega = np.square(corr_res)                       
                var1 = self._S1_approx(z, Omega)                  #S1_app estimate
                if estimate != 'S1':
                    mu = z.mean(axis=2)
                    nu_app = self._nu_approx(z, Omega)
                    var1 = self._S2_estim(mu, nu_app, var1)       #S2_app estimate
                    if estimate != 'S2' :
                        U_app = Omega.mean(axis = 1)    
                        V_app = self._V_approx(z, U_app)
                        var1 = self._S3_approx(mu, U_app, V_app, var1)  #S3_app estimate
      
            else :
                raise TypeError("Only booleans are allowed for the 'approx' argument")
                
            # Compute model variance induced by the random weights and the total variance
            var2 = self.y_var_/self.n_estimators
            var_predict = var1 + var2
            
        else :
            raise TypeError("Only 'S1', 'S2', or, 'S3' are allowed for the 'estimate' argument")
        
        return var_predict, noise_estimate    
    
class ELMEnsembleRidge(ELMEnsemble, BaseEstimator, RegressorMixin): 
    
    def __init__(self, n_estimators=20, n_neurons=100, alpha=1.0, 
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
            Regularization strength, aka Tikhonov factor. The default is 1.0.
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
        print("ELMEnsembleRidge __init__")      
        ELMEnsemble.__init__(self, n_estimators, n_neurons, activation, 
                            weight_distr, weight_scl, n_jobs, random_state)  
        self.alpha = alpha             

                
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
        print("ELMEnsembleRidge fit")                
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
            elm = ELMRidge(n_neurons = self.n_neurons,
                           alpha = self.alpha,
                           activation = self.activation,
                           weight_distr = self.weight_distr, 
                           weight_scl = self.weight_scl,
                           random_state = random_states[i])
            elm.fit(X,y)
            self.estimators_[i] = elm

        return self
  
    def _collect(self):
        '''
        Collect H_alpha_, H_predict and residuals in each model

        Returns
        -------
        H_alphas : numpy array of shape (n_neurons, n_samples, n_estimators)
            Pseudo-inverses of hidden matrices.
        H_predict : numpy array of shape (n_predicted_points, n_neurons, n_estimators)
            Hidden matrices for all the predicted points.
        residuals : numpy array of shape (n_samples, n_estimators)
            Training residuals for all models.
        '''
        print("ELMEnsembleRidge _collect") 
        n_obs = self.X_.shape[0]
        n_predict = self.X_predict_.shape[0]
        N = self.n_neurons
        M = self.n_estimators
        
        H_alphas = np.zeros((N, n_obs, M))
        H_predict = np.zeros((n_predict, N, M))
        residuals = np.zeros((n_obs, M))
        
        for i in range(M):   
            elm = self.estimators_[i]
            H_alphas[:, :, i] = elm.H_alpha_
            H_predict[:, :, i] = elm._H_compute(self.X_predict_)
            residuals[:, i] = elm.predict(self.X_) - self.y_
            
        return H_alphas, H_predict, residuals

    def homoskedastic_variance(self, approx=False):
        '''
        Compute an homoskedastic variance estimation of the model at last predicted points.
            
        Parameters
        ----------
        approx : bool, optional
            Speed up computation using the Hastie and Tibshirani approximation
            of effective degrees of freedom. Default is 'False'.

        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy.float
            Noise estimate.
            
        References
        ----------
        F. Guignard, F. Amato and M. Kanevski (in prep). On Extreme Learning 
        Machine Model Variance.
        
        T.J. Hastie, R.J. Tibshirani (1990). 
        Generalized additive models, CRC press. 
            
        ''' 
        print("ELMEnsembleRidge homoskedastic_variance")                 
        n_obs = self.X_.shape[0]
        H_alphas, H_predict, residuals = self._collect()
        Hs = self._collect_Hs()

        # Compute the effective degrees of freedom  
        Ptrace = np.einsum('ijm, jim -> m', Hs, H_alphas, optimize = 'greedy') # Traces of the Hat matrices 
        if approx == False :
            PPtrace = np.einsum('ijm, jkm, klm, lim -> m', Hs, H_alphas, Hs, H_alphas, 
                                optimize = 'greedy')  # Traces of the squared Hat matrices
            gamma = 2*Ptrace.mean() - PPtrace.mean()
            
        elif approx == True :
            gamma = -1/2 + 5/4*Ptrace.mean()        
            
        else:
            raise TypeError("Only booleans are allowed for the 'approx' argument")
        
        ddof = n_obs - gamma 
                
        # Compute noise estimation
        ARSS = self._AvgRSS(residuals)
        noise = ARSS / ddof
        
        #Compute model variance induced by noise
        muTmu = self._muTmu_estim(H_alphas, H_predict)
        var1 = noise * muTmu
        
        # Compute model variance induced by the random weights
        var2 = self.y_var_/self.n_estimators

        var_predict = var1 + var2 
            
        return var_predict, noise 
     
    

    
class ELMEnsembleRidgeCV(ELMEnsembleRidge, BaseEstimator, RegressorMixin):
    
    def __init__(self, n_estimators=20, n_neurons=100, alphas=np.array([0.1, 1.0, 10]), 
                 activation='logistic', weight_distr='uniform', weight_scl=1.0, 
                 n_jobs=1, random_state=None):
        '''
        Parameters
        ----------
        n_estimators : integer,
            Number of ELM models in the ensemble. The default is 20.
        n_neurons : integer,
            Number of neurons of each model. The default is 100.
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
        n_jobs : integer,
            Number of processors to use for the computation.
        random_state : integer, optional
            Random seed for reproductible results. The default is None.

        Returns
        -------
        None.

        '''        
        print("ELMEnsembleRidgeCV __init__")
        ELMEnsemble.__init__(self, n_estimators, n_neurons, activation, 
                             weight_distr, weight_scl, n_jobs, random_state)  
        self.alphas = alphas         
   
    
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
        print("ELMEnsembleRidgeCV fit")        
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
            elm = ELMRidgeCV(n_neurons = self.n_neurons,
                             alphas = self.alphas,
                             activation = self.activation,
                             weight_distr = self.weight_distr, 
                             weight_scl = self.weight_scl,
                             random_state = random_states[i])
            elm.fit(X,y)
            self.estimators_[i] = elm

        return self    


