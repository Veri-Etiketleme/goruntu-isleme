# Detecting Lane Lines on the Road

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project, we detect lane lines in videos using Python and OpenCV.  

## Project Instructions repo

The original Udacity project instructions can be found [here](https://github.com/udacity/CarND-LaneLines-P1).

## Coding environment

This project is coded using python and OpenCV in a Jupyter notebook. 

For installation instructions, please see [Udacity instructions](https://github.com/udacity/CarND-Term1-Starter-Kit).

## Overall steps

The method consists of the following steps:

* Loading each image from the video clip
* Grayscaling
* Gaussian smoothing
* Canny edge detection
* Region masking
* Applying Hough line transform
* Overlaying lanes on original image

## Detailed Writeup

Detailed report can be found in [_LaneDetection_writeup.md_](./LaneDetection_writeup.md).

## Solution video

![](./videos/solidWhiteRight_withDetectedLanes.mp4)

## Acknowledgments

I would like to thank Udacity for giving me this opportunity to work on an awesome project. Special thanks to Juan Marcos Ottonello for his article on "Extrapolating lines". 

