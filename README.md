# How can active learning solve the lack of labeled images for lung disease recognition

## Abstract
In this paper we'll use pool-based active machine learning
to try to reduce the amound of labeled images that are 
needed for lung disease recognition. We'll test and 
compare different query strategies, and experiment with 
enhancing the images.

## Setup
Make sure to you unzip all files in `/dataset` and `/enhanced`, the minimal directory structure should look like this:
```
.
├── dataset
│   ├── COVID
│   │   ├── images
│   │   └── masks
│   ├── Normal
│   │   ├── images
│   │   └── masks
│   └── Viral Pneumonia
│       ├── images
│       └── masks
├── enhanced
│   ├── COVID
│   │   ├── images
│   │   └── masks
│   ├── Normal
│   │   ├── images
│   │   └── masks
│   └── Viral Pneumonia
│       ├── images
│       └── masks
└── report.ipynb
```
Note that the `x.npy` and `y.npy` files are only there after running the initialization functions in the notebook.

### Contributors
Koen Desplenter, Quinten Vervynck

#### https://github.com/QuintenVervynck/mlal
