# **Finding Lane Lines on the Road** 

This project is about finding lane lines on a road.
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Draw lines on the found lanes as segments
* Draw lines on the found lanes as straight lines
* Reflect on my work on a written report

[//]: # (Image References)

[image2]: ./test_images/output.jpg "Output"

---

### Reflection

### 1. Pipeline

#### Finding Lane Lines ###
The first objective is to find the lane lines. Helpers functions are already here to help using openCV.
The first thing is to cread the image and to convert it to grayscale.
So we have something like this :
>gray = grayscale(image)

Then, we do the Gaussian Smoothing and the canny edge detection with the same parameters as in the class.

>kernel_size = 5

>blur_gray = gaussian_blur(image,kernel_size)

>low_threshold = 50

>high_threshold = 150

>edges = canny(blur_gray,low_threshold, high_threshold)

With this, we can define a region of interest, I used a polygon.
When we have the edges (from Canny), we can do the Hough space transformation.
There is only two parameters I want to talk about;

>min_line_length = 1

>max_line_gap = 1000

The first one says that there is no real minimum length for lines, any small segment can be considered as a line.
The second one helped me with the jitters from the yellowRightLane video that almost all disappeared by tuning this parameter.

>line_image = hough_lines(masked_image,rho,theta,threshold,min_line_length, max_line_gap)

Then we have the line_image on which we can apply the weighted_img function.
We save this as an output.
![alt text][image2]

#### Drawing on the road ####
We now have lane lines, the provided draw_lines function allows us to draw on the found segments and lines.
We now want to have straight lines on the road, and not segments.
Here's my approach :
* For everyline, we have (x,y) coordinates. The first thing is to calculate the slope

>slope = (y2-y1)/(x2-x1)

Since the image is reversed in terms of y, a negative slope would be a left line while a positive slope would be a right line.
I did not consider the case of the slope=0 in this exercise since the video clearly was not giving vertical lines.
Depending on the slope, we fill arrays of x and y and calculate min and max for these values. We then draw a line that goes from the bottom of the image to the vertice, with the slope.
We use the y=slope*x+intercept (slope and intercept are given by a polyfit function)
Finally, the openCV function **line** draws a line.
You can see the output in test_videos_output (the two videos solidWhiteRight and solidYellowLeft)


### 2. Identify potential shortcomings with your current pipeline

*As you can see in my P1 file, I did try to run the challenge video with my code and it was a disaster. This is probably due to the curve of the road.

*We still have few jitters on the yellow video that can disappear.

### 3. Suggest possible improvements to your pipeline

Many improvements are plausible :
* We could draw a region on which the vehicle is so we can detect its location.
* We could modify the vertices depending on the road; if we are on the right, it's a polygon, on the left, it has other dynamic coordinates... This way we could know if and when to switch position.

Thank you for reading.
