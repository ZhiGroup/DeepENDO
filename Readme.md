# Unsupervised representation learning of Brain MRI as phenotypes for genetic association studies
---
### This is the official repository accompanying the paper Xie et al. Unsupervised representation learning of Brain MRI as phenotypes for genetic association studies. 
---
### Overview
We use unsupervised learning based on 3D convolutional autoencoder architecture to derive 128-dimensional imaging derived endophenotypes to represent complex genetic architecture of the human brain. 

A seperate model is trained on T1 and T2. The model consists of an initial convolution block, four encoder blocks, a linear latent space of 128-dimension, four decoder blocks, and a final convolution block. Mean square error using a mask excluding background was used as loss. 

We also share our model weights at https://drive.google.com/drive/folders/16IXv-w6xpHhEQiSNjRSI8S5wS4QDjGKE?usp=sharing.  Please refer to prediction notebook in the interpretability file showing how to load the model weights.

### Data preprocessing
We did linear registration (12 DOF) to the preprocessed  brain extracted MRI provided by UKBiobank. 
https://git.fmrib.ox.ac.uk/falmagro/UK_biobank_pipeline_v_1/-/tree/master/

![Model architecture](files/Model_architecture.png)

For mapping genes identified through GWAS to the specific regions of brain, we used perturbation based approach. We add noise to the dimension of the interest in the endophenotype and then identify the changes observed in the reconstruction. 

![Interpretability](files/Interpretability.png)

### Code walkthrough

- [training directory](training) contains the files for running the training and instructions for data preparation. 
- [interpretability directory](interpretability) contains the files for running the interpretability 

### Dependencies
- [PyTorch 1.10.0](http://pytorch.org)
- [Nibabel 3.2.1](https://nipy.org/nibabel/)
- [Monai 0.7.0](https://monai.io/)
- [PyTorch lightning 1.4.9](https://www.pytorchlightning.ai/)
- [pandas 1.3.4](https://pandas.pydata.org/)
- [FSL 6.0.5 FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide)

### Reconstruction results

**Original T1 brain extracted MRI and reconstructed image from 128 dim latent space**
![Original T1 brain extracted MRI and reconstructed image from 128 dim latent space](files/Original_predicted_T1.png)


**Original T2 brain extracted MRI and reconstructed image from 128 dim latent space**
![Original T2 brain extracted MRI and reconstructed image from 128 dim latent space](files/Original_predicted_T2.png)

### How to Cite:


### Warning

* This repo is for research purpose. Using it at your own risk. 
* This repo is under GPL-v3 license. 
