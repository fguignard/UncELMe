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
        # print("ELMEnsemble __init__")
        self.n_estimators = n_estimators
        self.n_neurons = n_neurons
        self.activation = activation
        self.weight_distr = weight_distr
        self.weight_scl = weight_scl              
        self.n_jobs = n_jobs                        ## to parallelize
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
        
        Reference
        ---------
        G.-B. Huang, Q.-Y. Zhu, C.-K. Siew, 
        Extreme learning machine: theory and applications, 
        Neurocomputing 70 (1-3) (2006) 489–501.  
        
        '''    
        # print("ELMEnsemble fit")                
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
        # print("ELMEnsemble predict")                         
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
        # print("ELMEnsemble _collect") 
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
        # print("ELMEnsemble _AvgRSS") 
        sq_residuals = np.square(residuals)
        RSS = sq_residuals.sum(axis=0)
        ARSS = RSS.mean()
        
        return ARSS    
    
    def _muTmu_estim(self, H_pinvs, H_predict, estimate):
        '''
        Parameters
        ----------
        H_pinvs : Numpy array of shape (n_neurons, n_samples, n_estimators)
            Pseudoinverse matrices for all ELMs. (or H_alphas in the regularized case)
        H_predict : Numpy array of shape (n_predict, n_neurons, n_estimators)
            Hidden vectors at all predicted points for all ELMs.
        estimate : string, optional
            Estimate to use, "naive" or "bias-reduced".

        Returns
        -------
        muTmu : Numpy matrix of shape (n_predict)
            Unbiased estimate of the squared norm of mu
            
        '''    
        # print("ELMEnsemble _muTmu_estim") 
        N = self.n_estimators
        
        z = np.einsum('pnm, nsm -> psm', H_predict, H_pinvs, optimize = 'greedy')
        mu = z.mean(axis=2)
    
        muTmu = np.einsum('ps, ps -> p ', mu, mu)
        if estimate == 'bias-reduced':
            muTmu *= N/(N-1)
            quad_terms = np.einsum('psm, psm -> p ', z, z)
            muTmu += - quad_terms/(N*(N-1))
        elif estimate == 'naive' :
            None
        else :
            raise TypeError("Only 'bias-reduced' or 'naive' are allowed for the 'estimate' argument")
            
        return muTmu
    
    def homoskedastic_variance(self, estimate = 'bias-reduced'):
        '''
        Compute an homoskedastic variance estimation of the model at last predicted points.
            
        Parameters
        ----------
        estimate : string, optional
            Estimate to use, "naive" or "bias-reduced". The "bias-reduced" estimate 
            is highly recommended, see remark below. Default is "bias-reduced".
        
        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy.float
            Noise estimate.
            
        Remarks
        ------
        The "naive" estimate was proposed in a proceeding of the 28th European 
        Symposium on Artificial Neural Networks, Computational Intelligence and 
        Machine Learning (ESANN 2020). This work motivated the Neurocomputing 
        paper (see references below) in which the "bias-reduced" estimate was 
        proposed. In particular, it was shown that the later as a lower bias and, 
        therefore, was recommended. The "naive estimate" is still available for
        comparison and reproducibility purposes only.
            
        References
        ----------
        F. Guignard, F. Amato and M. Kanevski. 
        Uncertainty Quantification in Extreme Learning Machine:  
        Analytical Developments, Variance Estimates and Confidence Intervals, 
        Neurocomputing 456 (2021) 436-449.
        
        F. Guignard, M. Laib and M. Kanevski. 
        Model Variance for Extreme Learning Machine,
        ESANN (2020) 703-708.

        ''' 
        # print("ELMEnsemble homoskedastic_variance")                       
        n_obs = self.X_.shape[0]
        H_pinvs, H_predict, residuals = self._collect()
        
        # Compute noise estimation
        ddof = n_obs - self.n_neurons
        ARSS = self._AvgRSS(residuals)
        noise = ARSS / ddof
        
        #Compute model variance induced by noise
        muTmu = self._muTmu_estim(H_pinvs, H_predict, estimate)
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
        # print("ELMEnsemble _collect_Hs")  
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
        # print("ELMEnsemble _Sigma_estim")  
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
        # print("ELMEnsemble _S1_estim")  
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
        # print("ELMEnsemble _nu_estim") 
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
        # print("ELMEnsemble _S2_estim") 
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
        # print("ELMEnsemble _V_estim") 
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
        # print("ELMEnsemble _S3_estim") 
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
        # print("ELMEnsemble _S1_approx") 
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
        # print("ELMEnsemble _nu_approx") 
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
        # print("ELMEnsemble _V_approx") 
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
        # print("ELMEnsemble _S3_approx") 
        M = self.n_estimators
        muTUmu = np.einsum('ps, s, ps -> p', mu, U_app, mu, optimize = 'greedy')
        S3_app = M**2 * muTUmu - M*V_app - 2*(M-1)*S2_app
        S3_app = S3_app / ((M-1) * (M-2))
        
        return S3_app    
      
    def heteroskedastic_variance(self, estimate='S3', approx=False, corrected_residuals=False):
        '''
        Compute a variance estimation at last predicted points assuming 
        heteroskedasticity (non-constant noise variance).
            
        Parameters
        ----------
        estimate : string, optional
            Estimate to use, "naive", "S1, S2" or "S3". The "S3" estimate 
            is highly recommended, see remark below. Default is "S3".
        approx : bool, optional
            Speed up computation at the expense of a slightly approximate variance 
            estimation. Default is 'False'.
        corrected_residuals : bool, optional
            If True, the corrected residuals used to compute HC3 are squared and
            returned as noise estimates at training points. If False, the raw 
            residuals are used. Default is 'False'.
            
        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy array of shape (n_sample_train) 
            Noise estimate at the training points.

        Remarks
        ------
        The "naive" estimate was proposed in a proceeding of the 28th European 
        Symposium on Artificial Neural Networks, Computational Intelligence and 
        Machine Learning (ESANN 2020). This work motivated the Neurocomputing 
        paper (see references below) in which the "bias-reduced" estimate was 
        proposed. In particular, it was shown that the later as a lower bias and, 
        therefore, was recommended. The "naive estimate" is still available for
        comparison and reproducibility purposes only.
            
        References
        ----------
        F. Guignard, F. Amato and M. Kanevski. 
        Uncertainty Quantification in Extreme Learning Machine:  
        Analytical Developments, Variance Estimates and Confidence Intervals, 
        Neurocomputing 456 (2021) 436-449.
        
        F. Guignard, M. Laib and M. Kanevski. 
        Model Variance for Extreme Learning Machine,
        ESANN (2020) 703-708.
        
        J.G. MacKinnon, H. White. Some heteroskedasticity-consistent 
        covariance matrix estimators with improved finite sample properties, 
        Journal of econometrics 29 (3) (1985) 305--325. 
                    
        ''' 
        # print("ELMEnsemble heteroskedastic_variance")                                     
        H_pinvs, H_predict, residuals = self._collect()
        Hs = self._collect_Hs()
        
        z = np.einsum('pnm, nsm -> psm', H_predict, H_pinvs, optimize = 'greedy')
        Pdiag = np.einsum('nsm, snm -> nm', Hs, H_pinvs, optimize = 'greedy')
        corr_res = residuals/(1-Pdiag)
        
        # Noise estimate
        if corrected_residuals == False:
            noise_estimate = np.square(residuals).mean(axis=1)       
        elif corrected_residuals == True:
            noise_estimate = np.square(corr_res).mean(axis=1)
        else :
            raise TypeError("Only booleans are allowed for the 'corrected_residuals' argument")
        
        if estimate == 'naive' :   # naive estimation from ESANN proceeding
            mu = z.mean(axis=2)
            if approx == False :
                Sigma = self._Sigma_estim(corr_res)
                nu = self._nu_estim(z, Sigma)
            elif approx == True :
                Omega = np.square(corr_res)  
                nu = self._nu_approx(z, Omega)
            else :
                raise TypeError("Only booleans are allowed for the 'approx' argument")
            var1 = np.einsum('ps, ps -> p', mu, nu, optimize = 'greedy')                    
                
        elif estimate == 'S1' or estimate == 'S2' or estimate == 'S3': # estimation from Neurocomupting paper
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
        
        else :
            raise TypeError("Only 'S1', 'S2', 'S3' or 'naive' are allowed for the 'estimate' argument")
        
        # Compute model variance induced by the random weights and the total variance
        var2 = self.y_var_/self.n_estimators
        var_predict = var1 + var2
            
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
        # print("ELMEnsembleRidge __init__")      
        ELMEnsemble.__init__(self, n_estimators, n_neurons, activation, 
                            weight_distr, weight_scl, n_jobs, random_state)  
        self.alpha = alpha             

                
    def fit(self, X, y):
        '''
        Training for regularized ELM ensemble.
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
        
        Reference
        ---------
        W. Deng, Q. Zheng, L. Chen,  
        Regularized extreme learning machine, 
        IEEE symposium on computational intelligence and data mining, 2009, pp. 389–395.
        
        '''    
        # print("ELMEnsembleRidge fit")                
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
        # print("ELMEnsembleRidge _collect") 
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

    def homoskedastic_variance(self, estimate='bias-reduced'):
        '''
        Compute an homoskedastic variance estimation of the model at last predicted points.
            
        Parameters
        ----------
        estimate : string, optional
            Estimate to use, 'naive' or 'bias-reduced'. The 'bias-reduced' estimate 
            is recommended, see remark below. Default is 'bias-reduced'.

        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy.float
            Noise estimate.
 
        Remarks
        ------
        The "naive" estimate was proposed in a proceeding of the 28th European 
        Symposium on Artificial Neural Networks, Computational Intelligence and 
        Machine Learning (ESANN 2020). This work motivated the Neurocomputing 
        paper (see references below) in which the "bias-reduced" estimate was 
        proposed. In particular, it was shown that the later as a lower bias and, 
        therefore, was recommended. The "naive estimate" is still available for
        comparison and reproducibility purposes only.
            
        References
        ----------
        F. Guignard, F. Amato and M. Kanevski. 
        Uncertainty Quantification in Extreme Learning Machine:  
        Analytical Developments, Variance Estimates and Confidence Intervals, 
        Neurocomputing 456 (2021) 436-449.
        
        F. Guignard, M. Laib and M. Kanevski. 
        Model Variance for Extreme Learning Machine,
        ESANN (2020) 703-708.
            
        ''' 
        # print("ELMEnsembleRidge homoskedastic_variance")                 
        n_obs = self.X_.shape[0]
        H_alphas, H_predict, residuals = self._collect()
        Hs = self._collect_Hs()
        Hs = Hs.transpose(2,0,1)
        eigenHTHs = np.square(np.linalg.svd(Hs, full_matrices= False, compute_uv=False, hermitian=False))
        eigenP = (eigenHTHs/(eigenHTHs + self.alpha))
        eigenP2 = np.square(eigenP)
        self.gamma_ = (2*eigenP - eigenP2).sum() / self.n_estimators
        
        ddof = n_obs - self.gamma_ 
                
        # Compute noise estimation
        ARSS = self._AvgRSS(residuals)
        noise = ARSS / ddof
        
        #Compute model variance induced by noise
        muTmu = self._muTmu_estim(H_alphas, H_predict, estimate)
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
        # print("ELMEnsembleRidgeCV __init__")
        ELMEnsemble.__init__(self, n_estimators, n_neurons, activation, 
                             weight_distr, weight_scl, n_jobs, random_state)  
        self.alphas = alphas         
   
    
    def fit(self, X, y):
        '''
        Training for regularized ELM ensemble, with Generalized Cross Validation.
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
        
        References
        ----------
        W. Deng, Q. Zheng, L. Chen,  
        Regularized extreme learning machine, 
        IEEE symposium on computational intelligence and data mining, 2009, pp. 389–395.
        
        G. H. Golub, M. Heath, G. Wahba, 
        Generalized cross-validation as a method for choosing a good ridge parameter,
        Technometrics 21 (2) (1979) 215–223.
        
        '''    
        # print("ELMEnsembleRidgeCV fit")        
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
    
    def _collect_alphas_opt(self):
        '''
        Collect alpha_opt in each model. Complete the _collect() function for the
        homoskedastic GCV case.

        Returns
        -------
        alphas_opt : numpy array of shape (n_estimators)
             alpha_opt for each model of the ensemble.

        '''
        # print("ELMEnsembleRidgeCV _collect_alphas_opt")  
        M = self.n_estimators
        
        alphas_opt = np.zeros((M))  
        for i in range(M):   
            elm = self.estimators_[i]
            alphas_opt[i] = elm.alpha_opt  
            
        return alphas_opt

    def homoskedastic_variance(self, estimate='bias-reduced'):
        '''
        Compute an homoskedastic variance estimation of the model at last predicted points.
            
        Parameters
        ----------
        estimate : string, optional
            Estimate to use, 'naive' or 'bias-reduced'. The 'bias-reduced' estimate 
            is recommended, see remark below. Default is 'bias-reduced'.

        Returns
        -------
        var_predict : numpy array of shape (n_samples)
            Model variance estimation for y_predict.
        noise_estimate : numpy.float
            Noise estimate.
 
        Remarks
        ------
        The "naive" estimate was proposed in a proceeding of the 28th European 
        Symposium on Artificial Neural Networks, Computational Intelligence and 
        Machine Learning (ESANN 2020). This work motivated the Neurocomputing 
        paper (see references below) in which the "bias-reduced" estimate was 
        proposed. In particular, it was shown that the later as a lower bias and, 
        therefore, was recommended. The "naive estimate" is still available for
        comparison and reproducibility purposes only.
            
        References
        ----------
        F. Guignard, F. Amato and M. Kanevski. 
        Uncertainty Quantification in Extreme Learning Machine:  
        Analytical Developments, Variance Estimates and Confidence Intervals, 
        Neurocomputing 456 (2021) 436-449.
        
        F. Guignard, M. Laib and M. Kanevski. 
        Model Variance for Extreme Learning Machine,
        ESANN (2020) 703-708.
            
        ''' 
        # print("ELMEnsembleRidgeCV homoskedastic_variance")                 
        n_obs = self.X_.shape[0]
        H_alphas, H_predict, residuals = self._collect()
        Hs = self._collect_Hs()          
        Hs = Hs.transpose(2,0,1)
        alphas_opt = self._collect_alphas_opt()                                 ### change from non GCV case
        alphas_opt = alphas_opt.reshape((alphas_opt.shape[0], 1))               ### change from non GCV case
        alphas_opt = alphas_opt.repeat(self.n_neurons, axis = 1)                ### change from non GCV case
        eigenHTHs = np.square(np.linalg.svd(Hs, full_matrices= False, compute_uv=False, hermitian=False))
        eigenP = (eigenHTHs/(eigenHTHs + alphas_opt))                           ### change from non GCV case
        eigenP2 = np.square(eigenP)
        self.gamma_ = (2*eigenP - eigenP2).sum() / self.n_estimators
        
        ddof = n_obs - self.gamma_ 
                
        # Compute noise estimation
        ARSS = self._AvgRSS(residuals)
        noise = ARSS / ddof
        
        #Compute model variance induced by noise
        muTmu = self._muTmu_estim(H_alphas, H_predict, estimate)
        var1 = noise * muTmu
        
        # Compute model variance induced by the random weights
        var2 = self.y_var_/self.n_estimators

        var_predict = var1 + var2 
            
        return var_predict, noise 

# Future improvement
# This homoskedastic_variance function could be re-written here : 
# 1) Indeed, the svd is done twice when this class is used, because the homoskedastic_variance 
#    function is inherited from the class ELMEnsembleRidge, where the svd is not done 
#    during the fitting process. It could be done one time and store during the 
#    fitting process of the ELMEnsembleRidgeCV and the ELMRidgeCV classes (elm_estimator.py).
# 2) The writing could be optimize, as there is not too much changes comparing to 
#    the ELMEnsembleRidge corresponding function.        
# 3) Eventually put the alphas_opt in atribute during the fitting process.
