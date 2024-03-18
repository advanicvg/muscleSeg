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
* opencv-python>=4.5
* pydicom
* numpy
* tqdm

## How to run the code
Simply excute the command line script:
`python main.py`
