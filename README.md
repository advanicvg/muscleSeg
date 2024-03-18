## Introduction
In this project, we create a DNN model for muscles segmentaion using supervised learning. The input is any abdominal computed tomography (CT), at L3 level, and the output is the segmentation of muscles regions and L3 vertebral body, namely,

* rectus abdominus muscle
* trans abd,int and ext obl
* psoas major mucle
* quardratus lumborum muscle
* eretor spinae muscle
* L3 Vertebral body

## Prerequirement
To run the code, one needs to install the following python dependancies first:

* torch>=1.7
* torchvision
* albumentations==0.5
* segmentation-models-pytorch
* [Ranger-Deep-Learning-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
* opencv-python>=4.5
* pydicom
* numpy
* tqdm

## Data Arrangement
The format of input data are DICOM and its annotations(masks) are saved as .npy files. The training dataset and validation dataset are stored in forlde `./train` and `./valid` respectively.

## How to run the code
Simply excute the command line script:
`python main.py`
