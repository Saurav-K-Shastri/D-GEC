# D-GEC

This repository containts the code associated with the paper "Denoising Generalized Expectation-Consistent Approximation for MR Image Recovery," by Saurav K. Shastri, Rizwan Ahmad, Christopher A. Metzler, and Philip Schniter

arXiv paper Link: https://arxiv.org/pdf/2206.05049.pdf

### Abstract

To solve inverse problems, plug-and-play (PnP) methods replace the proximal step in a convex optimization algorithm with a call to an application-specific denoiser, often implemented using a deep neural network (DNN).  Although such methods yield accurate solutions, they can be improved.  For example, denoisers are usually designed/trained to remove white Gaussian noise, but the denoiser input error in PnP algorithms is usually far from white or Gaussian.  Approximate message passing (AMP) methods provide white and Gaussian denoiser input error, but only when the forward operator is sufficiently random.  In this work, for Fourier-based forward operators, we propose a PnP algorithm based on generalized expectation-consistent (GEC) approximation---a close cousin of AMP---that offers predictable error statistics at each iteration, as well as a new DNN denoiser that leverages those statistics.  We apply our approach to magnetic resonance (MR) image recovery and demonstrate its advantages over existing PnP and AMP methods. 

### Demo
The Jupyter Notebook "Fast_DGEC_vs_PnP_Demo_Notebook.ipynb" contains demo of the multicoil fastMRI image reconstruction using the D-GEC algorithm. This demo compares the performance of D-GEC and PnP-PDS. 

The Jupyter Notebook "Example_DGEC_behavior.ipynb" contains an example of D-GEC behavior in multicoil MRI with a 2D line mask.

### Dependencies

Please download the data and the pre-trained denoisers required for the demo here: https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/shastri_19_buckeyemail_osu_edu/EpXV2uwXrJxArgnizYz7OMMBHw3YxOYugjG3JWj6jnmYCw?e=xVazha

To make sure that the versions of the packages match those we tested the code with, we recommend creating a new virtual environment and installing packages using `conda` with the following command.

```bash
conda env create -f environment.yml
```


