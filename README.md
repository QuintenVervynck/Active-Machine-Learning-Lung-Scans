# How can active learning solve the lack of labeled images for lung disease recognition

### Authors
Koen Desplenter, Quinten Vervynck

#### https://github.com/QuintenVervynck/mlal

## Abstract
In this paper we'll use pool-based active machine learning
to try to reduce the amount of labeled images that are 
needed for lung disease recognition. We'll test and 
compare different query strategies, and experiment with 
enhancing the images.

## Setup
Make sure to you unzip all files in `/dataset` and `/enhanced` (some are 7z, watch out).

The minimal directory structure needed to run the notebook should look like this:
```
.
├── dataset
│   ├── COVID
│   │   ├── images
│   │   └── masks
│   ├── Normal
│   │   ├── images
│   │   └── masks
│   ├── Viral Pneumonia
│   │   ├── images
│   │   └── masks
│   ├── x.npy
│   └── y.npy
├── enhanced
│   ├── COVID
│   │   ├── images
│   │   └── masks
│   ├── Normal
│   │   ├── images
│   │   └── masks
│   ├── Viral Pneumonia
│   │   ├── images
│   │   └── masks
│   ├── x.npy
│   └── y.npy
└── report.ipynb
```
Note that because the `x.npy` and `y.npy` files are already here, you do not need to run any of the initial setup functions (which take a long time if you do decide to run them).


