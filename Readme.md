# [New phenotype discovery method by unsupervised deep representation learning empowers genetic association studies of brain imaging](https://www.medrxiv.org/content/10.1101/2022.12.10.22283302v1)
---
This is the official repository accompanying the paper.

## Authors: 

<h4> Khush Patel, Ziqian Xie, Hao Yuan, Sheikh Muhammad Saiful Islam, Wanheng Zhang, Assaf Gottlieb, Han Chen, Luca Giancardo, Alexander Knaack, Evan Fletcher, Myriam Fornage, Shuiwang Ji, Degui Zhi. </h4>

<hr />

## Overview
We use unsupervised learning based on 3D convolutional autoencoder architecture to derive 128-dimensional imaging derived endophenotypes to represent complex genetic architecture of the human brain. The deep learning derived endophenotypes (ENDOs) identify 1,132 significant (P<5*10-8/256) SNP-ENDO pairs, out of which 658 are replicated (P<0.05/1132) in a seperate replication cohort.

**Overall Pipeline**

<img src="files/Overall_figure.jpg" width=800 align="center">

**Aggregated Miami plot of all 256 single ENDO GWASs in discovery and replication cohorts**

<img src="files/miami_plot.jpg" width=800 align="center">

A seperate model is trained on T1 and T2. The model consists of an initial convolution block, four encoder blocks, a linear latent space of 128-dimension, four decoder blocks, and a final convolution block. Mean square error using a mask excluding background was used as loss. 

We also share our model weights at https://drive.google.com/drive/folders/16IXv-w6xpHhEQiSNjRSI8S5wS4QDjGKE?usp=sharing.  Please refer to prediction notebook in the interpretation file showing how to load the model weights.

**Model architecture**

<img src="files/Model_architecture.jpg" width=800 align="center">

For mapping genes identified through GWAS to the specific regions of brain, we used perturbation based approach. We add noise to the dimension of the interest in the endophenotype and then identify the changes observed in the reconstruction. 

**Interpretation pipeline**

<img src="interpretation/Interpretability.jpg" width=800 align="center">


## Code walkthrough

- [training directory](training) contains the files for running the training and instructions for data preparation. 
- [interpretation directory](interpretation) contains the files for running the decoder generated perturbation based approach. 

## Dependencies
- [PyTorch 1.10.0](http://pytorch.org)
- [Nibabel 3.2.1](https://nipy.org/nibabel/)
- [Monai 0.7.0](https://monai.io/)
- [PyTorch lightning 1.4.9](https://www.pytorchlightning.ai/)
- [pandas 1.3.4](https://pandas.pydata.org/)
- [torchmetrics 0.8.2](https://torchmetrics.readthedocs.io/en/stable/)
- [FSL 6.0.5 FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide)

## Data preprocessing
We did linear registration (12 DOF) to the preprocessed  brain extracted MRI provided by UKBiobank. 
https://git.fmrib.ox.ac.uk/falmagro/UK_biobank_pipeline_v_1/-/tree/master/

## Reconstruction results

**Original T1 brain extracted MRI and reconstructed image from 128 dim latent space**



**Original T2 brain extracted MRI and reconstructed image from 128 dim latent space**
<img src="files/T2_lightbox.png" width=800 align="center">


## Acknowledgements

This work was supported by grants from the National Institute of Aging (U01 AG070112-01A1). In addition, L.G. is supported in part by NIH grants UL1TR003167 and R01NS121154.

## How to Cite:

Patel, K., Xie, Z., Yuan, H., Islam, S. M. S., Zhang, W., Gottlieb, A., Chen, H., Giancardo, L., Knaack, A., Fletcher, E., Fornage, M., Ji, S., & Zhi, D. (2022). New phenotype discovery method by unsupervised deep representation learning empowers genetic association studies of brain imaging. MedRxiv, 2022.12.10.22283302. https://doi.org/10.1101/2022.12.10.22283302

## Warning

* This repo is for research purpose. Using it at your own risk. 

