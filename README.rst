=====================================
UncELMe
=====================================

Python package for Uncertainty quantification of Extreme Learning Machine ensemble.

UncELMe contains :

* The ELM, ELMRidge and ELMRidgeCV classes, which are scikit-learn compatible estimators for regression based on Extreme Learning Machine (ELM), with regularization possibility (ridge estimate).

* The ELMEnsemble, ELMEnsembleRidge and ELMEnsembleRidgeCV classes, which allows ensemble of ELM, ELMRidge and ELMRidgeCV estimators.

* Estimates of model variance for the ensemble, including homoskedastic and heteroskedastic estimates for the  non-regularized and regularized cases.

More theoretical and implementation details can be found in 

> Guignard et al.

> Neurocomputing
> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 

> <https://doi.org/10.1016/j.neucom.2021.04.027>. Please refer to this article if you are using the package.


The package can be installed via pip install command.

* License : MIT



Require
--------

* numpy: 1.17.4
* sklearn: 0.21.3
