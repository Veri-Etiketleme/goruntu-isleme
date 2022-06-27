
# Detecting Lane Lines on the Road 

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project, we detect lane lines in videos using Python and OpenCV.  

## Overall steps

The method consists of the following steps:

* Loading each image from the video clip
* Grayscaling
* Gaussian smoothing
* Canny edge detection
* Region masking
* Applying Hough line transform
* Overlaying lanes on original image

## Lane detection methodology

The method consists of the following steps:

### 1) Loading each image from the video clip

The method works on each image from the video clip. In the end, the processed images are framed together to generate a new 
video clip with the detected lanes superimposed on each frame of the video. It is always a good idea to see the type of input image and its respective dimensions. Plotting the image illustrates whether the image is square or rectangular. The original image is as follows:

[image1]: ./figures/original_image.jpg "Original image"
![alt text][image1]

The image is of type numpy array with dimensions: (540, 960, 3).

### 2) Grayscaling

The input images are in RGB format. Many of the computer vision algorithms such as Canny edge detector operate on grayscaled images. The input image is converted into a grayscaled image using `cvtColor(img, COLOR_RGB2GRAY)` function from OpenCV. The grayscaled image is depicted as follows:

[image2]: ./figures/grayscaled_image.jpg "Grayscaled image"
![alt text][image2]

*img* is the input RGB image.  

### 3) Gaussian smoothing

As a preprocessing step to Canny edge detection, the grayscaled image is smoothed using Gaussian blurring method using `GaussianBlur(img, (kernel_size, kernel_size), 0)` function from OpenCV. Though the Canny edge detection includes Gaussian blurring, doing it apriori allows flexibility to choose different filter kernel sizes. We used the *kernel size* of 5. The Gaussian smoothed image is depicted as follows:

[image3]: ./figures/guassian_filtered_image.jpg "Gaussian blurred image"
![alt text][image3]

*img* is the grayscaled image.

### 4) Canny edge detection

Edges are detected on the Gaussian smoothed image using the Canny edge detector with the OpenCV function `Canny(img, low_threshold, high_threshold)`. Following edges were detected:

[image4]: ./figures/canny_edge_image.jpg "Canny edges"
![alt text][image4]

*img* is the Gaussian smoothed *image*, *low threshold = 50*, and *high threshold = 150*.

### 5) Region masking

As seen in the above step, edges are detected all over the image, wherever there is a significant gradient change. To restrict the edges to specific region of an image where the lanes are most likely present, region mask is found using the function `fillPoly(mask, vertices, ignore_mask_color)` from OpenCV. The output of this step is depicted as follows:

[image5]: ./figures/region_mask.jpg "Region mask"
![alt text][image5]

*mask* is a blank image, *vertices* indicate the x and y coordinates of the edge points to which the polygon is fit, and *ignore_ mask_color* = 255 is the color used to fill the mask.

The region mask is applied to the edges using `bitwise_and(img, mask)` function from OpenCV. The output of this step is as shown below:

[image6]: ./figures/masked_edges.jpg "Masked edges"
![alt text][image6]

*img* is the image with edges.

### 6) Hough line transform

Canny edge detector finds edges as points which are strewn together and converted into line segments using Hough transform. The function `HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)` from OpenCV runs probabilisitc Hough transform on the canny edges. Probabilistic Hough Transform is an optimization of Hough Transform that does not take all the points into consideration, instead take only a random subset of points and that is sufficient for line detection.The input arguments to the **HoughLinesP()** function are as follows:

* rho = 1 (distance resolution in pixels of the Hough grid)
* theta = pi/180 (angular resolution in radians of the Hough grid)
* threshold = 30 (minimum number of votes (intersections in Hough grid cell))
* minLineLength = 10 (minimum number of pixels making up a line; line segments shorter than this are rejected)
* maxLineGap = 1 (maximum allowed gap between line segments to treat them as single line)

The function outputs _x_ and _y_ coordinates of the start and end of line segments. The OpenCV function `line(img, (x1, y1), (x2, y2), color, thickness)` draws these segments on the image inplace. As can be seen below, the line segments constituting the left and right lanes are broken.

[image7]: ./figures/hough_lines.jpg "Line image"
![alt text][image7]

The broken line segments are stitched together to form a consistent lane in the following steps:

1. Separate the line segments into left and right line segments based on the sign of their slopes. Only the slope values greater than 'slope_thresh' are retained by getting rid of slope values not corresponding to lanes. This helps removing spurious slope values left out after the region masking step.
    
2. Calculate the average segment by calculating the average values of each co-ordinate from the segments on left and right side. In other words, calculate the average of x1 values (x1_avg), x2 values (x2_avg), y1 values (y1_avg), and y2 values (y2_avg). Calculate the slope and intercept of the average segment formed by (x1_avg, y1_avg) and (x2_avg, y2_avg).
    
3. Find the start and end points on each side using the above average slope and intercept values, image size, and end point of region of interest polygon. 
    
4. Draw lines joining the start and end points on each side with given color and thickness. 

The averaged and extrapolated line segments depicting the lanes can be seen below:

[image8]: ./figures/hough_lines_ext.jpg "Line image with extrapolated lanes"
![alt text][image8]

### 7) Overlaying lanes on original image

The detected lanes are overlayed on the original image using the function `addWeighted(initial_img, α, img, β, γ)` from OpenCV. The final image is computed as follows: α.initial_img + β.img + γ, and can be seen below:

[image9]: ./figures/detected_lanes_ext.jpg "Original image with superimposed extrapolated lines"
![alt text][image9]

*initial_img* is the blank image with averaged and extrapolated line segments depicting the lanes, and *img* is the original image. *α*, *β*, and *γ* are the weights. 




```python
# <video controls src="videos/solidWhiteRight_withDetectedLanes.mp4" />
```


```python
# <video controls src="videos/solidYellowLeft_withDetectedLanes.mp4" />
```


```python
# <video controls src="videos/challenge_withDetectedLanes.mp4" />
```

## Shortcomings of the proposed approach

The method assumes a lighting condition such as bright sunlight where the lanes are clearly visible and have very different color and intensity compared to most of the rest of the background. The method also assumes that the lanes are not winding in opposite direction. In other words, the lanes may start as parallel and appear to converge over distance in the image, but they will not cross each other repeatedly in which case the concept of left and right is reversed. If the above assumptions are violated, the method will most likely fail.

## Possible improvements

The method must be made robust to lighting conditions by applying a transform such as homomorphic filtering which is robust to illumination changes. Transforming the image to be in infrared spectrum might also help as is typically used in military applications to identify criminals in pitch black environment. 

### Acknowledgments

I would like to thank Udacity for giving me this opportunity to work on an awesome project. Special thanks to Juan Marcos Ottonello for his article on "Extrapolating lines". 

