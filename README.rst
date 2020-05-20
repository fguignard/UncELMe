=====================================
UncELMe
=====================================

Python package for Uncertainty quantification of Extreme Learning Machine ensemble.

So far, UncELMe contains :

* The ELMRegressor class, which is a scikit-learn compatible estimator for regression based on Extreme Learning Machine (ELM), with regularization possibility (ridge estimate).

* The ELMEnsemble class, which allows ensemble of ELMRegressor.

* Estimates of model variance for the ensemble. This part is in progress and contains for the moment naive estimates for the homoskedastic and heteroskedastic non-regularized cases.


The package can be installed via pip install command.

* License : MIT



Require
--------

* numpy: 1.17.4
* sklearn: 0.21.3
