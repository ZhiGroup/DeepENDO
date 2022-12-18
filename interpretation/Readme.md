# Code files for generating interpretability for the endophenotype.

## The basic principle behind interpretability of a particular endophenotype is to perturb that specific endophenotype by adding noise (while keeping other ENDOs constant) and identifying corresponding changes over the MRI. 

<img src="Interpretability.jpg" width=800 align="center">

### Code walkthrough

- perturbation based approach.ipynb: Perturbation based approach using decoder to map ENDOs to regions of brain. 

- KS_statistic.ipynb: Generate KS statistic plots using Harvard Oxford cortical and subcortical atlas. Harvard Oxford subcortical atlas includes labels cerebral cortex and white matter which are included here but can be excluded as in manuscript figures to emphasize subcortical structures.

- model128.py: contains the autoencoder model used for the training.

- ENDOs_T2_128.ipynb: Sample code to generate 128 dimensional ENDOs

### Requirements

- You will need to generate the 128 dim ENDOs from the T1 and T2 MRI stored in your local directory to use in jupyter notebook above. Sample code to do the same is provided.
