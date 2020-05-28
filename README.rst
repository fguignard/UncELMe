=====================================
UncELMe
=====================================

Python package for Uncertainty quantification of Extreme Learning Machine ensemble.

So far, UncELMe contains :

* The ELM, ELMRidge and ELMRidgeCV classes, which are scikit-learn compatible estimators for regression based on Extreme Learning Machine (ELM), with regularization possibility (ridge estimate).

* The ELMEnsemble, ELMEnsembleRidge and ELMEnsembleRidgeCV classes, which allows ensemble of ELM, ELMRidge and ELMRidgeCV estimators.

* Estimates of model variance for the ensemble. This part is in progress and contains for the moment naive estimates for the homoskedastic and heteroskedastic non-regularized cases.


The package can be installed via pip install command.

* License : MIT



Require
--------

* numpy: 1.17.4
* sklearn: 0.21.3
