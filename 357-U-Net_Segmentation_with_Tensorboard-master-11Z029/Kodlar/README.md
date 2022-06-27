# U-Net Segmentation with Tensorboard

![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTooTouch%2FU-Net_Segmentation_with_Tensorboard)

This is a simple implementation of the U-Net arhitecture and a project to utilize segmentation using Tensorboard.

A Korean translation of U-Net Paper: http://bit.ly/UNet_Paper_Translation

This tutorial depends on the following libraries:

- Tensorflow == 1.14  
- Keras == 2.3.1

My computing resources are as follows:

- CPU: Intel i7-8700k  
- GPU: GTX 1080ti  
- RAM: 64GB  

# How to Run
```
python main.py
```

**Tensorboard**
```
tensorboard --logdir=./logs --host localhost
```


---
# Overview

## Data
The original dataset is from [ISBI challenge](http://brainiac2.mit.edu/isbi_challenge/home), and I've downloaded it and done the pre-processing.

You can find it in folder data/membrane.

## Model Architecture
![](https://github.com/bllfpc/U-Net_Segmentation/blob/master/images/u-net-architecture.png)

## Training Detail
1. Data Augmentation

10 times more images were used from the original number.

Method | Value 
---|---
Rotation Range | 0.2
Width Shift Range | 0.05
Height Shift Range | 0.05
Shear range | 0.05
Zoom range | 0.05
Horizontal Flip | True
Fill Mode | reflect

**Augmentation images examples**
![](https://github.com/bllfpc/U-Net_Segmentation/blob/master/images/augmentation.png)

2. Hyperparameters
- epochs : 50
- batch size : 5
- Learning rate : 0.0001

3. Optimizer, Loss function and Metric
- Adam
- Binary Cross Entropy
- Accuracy

## Results

Model performance was approximately 91% accuracy for validation data when 50 epochs were trained.

![](https://github.com/bllfpc/U-Net_Segmentation/blob/master/images/results.png)

## Tensorboard
![](https://github.com/bllfpc/U-Net_Segmentation/blob/master/images/tensorboard.gif)

# Reference
- Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- Code: https://github.com/zhixuhao/unet
