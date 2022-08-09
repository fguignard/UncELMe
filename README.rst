UncELMe
=====================================

Python package for Uncertainty quantification of Extreme Learning Machine ensemble.

UncELMe contains :

* The ELM, ELMRidge and ELMRidgeCV classes, which are scikit-learn compatible estimators for regression based on Extreme Learning Machine (ELM), with regularization possibility (ridge estimate).

* The ELMEnsemble, ELMEnsembleRidge and ELMEnsembleRidgeCV classes, which allows ensemble of ELM, ELMRidge and ELMRidgeCV estimators.

* Estimates of model variance for the ensemble, including homoskedastic and heteroskedastic estimates for the  non-regularized and regularized cases.

More theoretical and implementation details can be found in [Guignard et al.](https://doi.org/10.1016/j.neucom.2021.04.027). Please refer to this article if you are using the package.

Examples with the ELMEnsemble class are availaible on GitHub Gist to help you get started :

* A [one-dimensional simulated case](https://gist.github.com/fguignard/e830899621ba79b2b6fb97e9f0d82ccb) using homoskedastic estimate.

* An [example with the Boston data set](https://gist.github.com/fguignard/fc590de1b82621ed58fd82b9ef37a4fd) using homoskedastic and heteroskedastic estimates.

Examples with ELMEnsembleRidge and ELMEnsembleRidgeCV classes will follow.

The package can be installed via pip install command.

```pip install UncELMe```

License : MIT

