import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def camcal(imagesPath, nx=9, ny=6):
    """
    Generate camera calibration metrics using the provided set of calibration images
    """    
    # Make a list of calibration images
    images = glob.glob(imagesPath)
     
    objpts = []
    imgpts = []
     
    # create array of vertice target points - this is the same for all calibration images 
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
     
    for fname in images:
        img = cv2.imread(fname)     # read image data from file
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)  # Find the chessboard corners
        # If found, collect data for calibration
        if ret == True:
            imgpts.append(corners)
            objpts.append(objp)
             
    # generate and return calibaration metrics
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None) 
    return(mtx, dist) 

def inspect1(src, dst):
    """
    Output 2 images for comparison.
    
    These can be any two images but typically the first is the original and the 2nd is processed.
    """    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    f.tight_layout()
    ax1.imshow(src, cmap='gray')
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(dst, cmap='gray')
    ax2.set_title('Processed Image', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
def inspect3(src, dst1, dst2, dst3):
    """
    Output 4 images for comparison.
    
    These can be any four images but typically the first is the original and the rest are processed.
    """    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 5))
    f.tight_layout()
    ax1.imshow(src)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(dst1, cmap='gray')
    ax2.set_title('Processed Image 1', fontsize=10)
    ax3.imshow(dst2, cmap='gray')
    ax3.set_title('Processed Image 2', fontsize=10)
    ax4.imshow(dst3, cmap='gray')
    ax4.set_title('Processed Image 3', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
def checkcal(img, mtx, dist):
    """
    For visual inspection to validate the correctness of camera calibration metrics.
    """    
    src = mpimg.imread(img)
    dst = cv2.undistort(src, mtx, dist, None, mtx)
    inspect1(src,dst)    

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    """
    # region of interest vertices
    region_vertices = np.array([[(571,440),
                                 (709,440),
                                 (1200,720),
                                 (920,720),
                                 (640,480),
                                 (360,720),
                                 (80,720)]], np.int32)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    mask = np.zeros_like(img)       #defining a blank mask to start with
    cv2.fillPoly(mask, region_vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Apply sobel x and y, compute the direction of the gradient and apply a threshold.
    """    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.sqrt(sobel_x**2)
    abs_sobel_y = np.sqrt(sobel_y**2)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    angle = np.arctan2(abs_sobel_y, abs_sobel_x)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(angle)
    binary_output[(angle >= thresh[0]) & (angle <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

def sobel_x(img, sobel_kernel=3):
    """
    Apply sobel x to image.
    """    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return scaled_sobel

def sobel_y(img, sobel_kernel=3):
    """
    Apply sobel y to image.
    """    
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    return scaled_sobel

def threshold(img, thresh=(20,100)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary
             
def inv_threshold(img, thresh=(20,100)):
    binary = np.ones_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 0
    return binary

def and_binaries(bin1, bin2): 
    combined_binary = np.zeros_like(bin1)
    combined_binary[(bin1 == 1) & (bin2 == 1)] = 1
    return combined_binary
             
def combine_binaries(bin1, bin2): 
    combined_binary = np.zeros_like(bin1)
    combined_binary[(bin1 == 1) | (bin2 == 1)] = 1
    return combined_binary
             
def warp_transforms(img_size):
    """
    compute the warp and inverse warp transforms from the passed vertices.
    """    
    src_vertices = np.array([[(571,461),
                              (709,461),
                              (1030,660),
                              (250,660)]], np.float32)
    dst_vertices = np.array([[(250,0),
                              (1030,0),
                              (1030,660),
                              (250,660)]], np.float32)

    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    #Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(dst_vertices, src_vertices)
    
    return M, Minv

def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def pipeline(img):
    img = np.copy(img)
   
    # Apply threshold on r channel
    r_channel = img[:,:,0]
    r_binary = threshold(r_channel, thresh=(217,255))
    
    # extract h and l channel info from HLS encoding
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]

    # Apply threshold to h channel to extract yellow hue (lane line color)
    y_binary = threshold(h_channel, thresh=(18,24))
    
    # Apply sobel x to l channel
    lx = sobel_x(l_channel, sobel_kernel=3)
    lx_binary = threshold(lx, thresh=(50,90))

    # Extract V channel info (from HSV encoding) and apply threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    V_channel = hsv[:,:,2]
    V_binary = threshold(V_channel, thresh=(180,215))
    
    # Apply sobel x on Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = sobel_x(gray) # Take the derivative in x
    gx_binary = threshold(gx, thresh=(20,100))
    
    # Combine the all the binary thresholded layers extracted above and return
    combined_binary = np.zeros_like(r_binary)
    combined_binary = combine_binaries(combined_binary, r_binary)
    combined_binary = combine_binaries(combined_binary, lx_binary)
    combined_binary = combine_binaries(combined_binary, V_binary)
    combined_binary = combine_binaries(combined_binary, gx_binary)
    combined_binary = combine_binaries(combined_binary, y_binary)

    return combined_binary

# Line class receives the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.fit = [np.array([False])]  
        self.fit_cr = [np.array([False])]  
        # last successful polynomial line fit
        self.last_fit = [np.array([False])]
        self.last_fit_cr = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        # x values of most recent fit
        self.fitx = None
        # number of attempts since the last successful fit
        self.no_fit_count = 0

left = Line()
right = Line()
def fit_lines(binary_warped):
    global left
    global right
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit a second order polynomial to each set of lane pixels.
    # Do the fit in pixels and then again in meters for real world space
    # If no lane points were found use last successful fit from previous frame.
    if leftx.shape[0] > 0:
        left.fit = np.polyfit(lefty, leftx, 2)
        left.fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        left.no_fit_count = 0
    else:
        left.fit = left.last_fit    # use last successful fit from previous frame
        left.fit_cr = left.last_fit_cr
        left.no_fit_count += 1
        
    if rightx.shape[0] > 0:
        right.fit = np.polyfit(righty, rightx, 2)
        right.fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        right.no_fit_count = 0
    else:
        right.fit = right.last_fit  # use last successful fit from previous frame
        right.fit_cr = right.last_fit_cr
        right.no_fit_count += 1
        
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty) * ym_per_pix

    # Calculate radius of curvature, in meters, of the detected lines
    left.radius_of_curvature = ((1 + (2*left.fit_cr[0]*y_eval + left.fit_cr[1])**2)**1.5) / np.absolute(2*left.fit_cr[0])
    right.radius_of_curvature = ((1 + (2*right.fit_cr[0]*y_eval + right.fit_cr[1])**2)**1.5) / np.absolute(2*right.fit_cr[0])

    # Add sanity check of the lines here.  If any sanity check fails revert to last good frame
    # - Check that they have similar curvature
    # - TODO: Check that they are separated by approximately the right distance horizontally
    # - TODO: Check that they are roughly parallel
    if (((right.radius_of_curvature/left.radius_of_curvature) < 0.1)
        | ((right.radius_of_curvature/left.radius_of_curvature) > 10.0)):
#        print("\r", left.radius_of_curvature,right.radius_of_curvature)
        left.fit = left.last_fit
        left.fit_cr = left.last_fit_cr
        left.radius_of_curvature = ((1 + (2*left.fit_cr[0]*y_eval + left.fit_cr[1])**2)**1.5) / np.absolute(2*left.fit_cr[0]) 
        right.fit = right.last_fit
        right.fit_cr = right.last_fit_cr
        right.radius_of_curvature = ((1 + (2*right.fit_cr[0]*y_eval + right.fit_cr[1])**2)**1.5) / np.absolute(2*right.fit_cr[0]) 
           
    # Generate x and y values for plotting
    left.fitx = left.fit[0]*ploty**2 + left.fit[1]*ploty + left.fit[2]
    right.fitx = right.fit[0]*ploty**2 + right.fit[1]*ploty + right.fit[2]

    # Save last good fit data
    left.last_fit = left.fit
    left.last_fit_cr = left.fit_cr
    right.last_fit = right.fit
    right.last_fit_cr = right.fit_cr

    # show intermediate result if desired
    if 0:    
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
 
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left.fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left.fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([fight.fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fight.fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
      
        left_line_line1 = np.array([np.transpose(np.vstack([left.fitx-1, ploty]))])
        left_line_line2 = np.array([np.flipud(np.transpose(np.vstack([left.fitx+1, ploty])))])
        left_line_line_pts = np.hstack((left_line_line1, left_line_line2))
        right_line_line1 = np.array([np.transpose(np.vstack([fight.fitx-1, ploty]))])
        right_line_line2 = np.array([np.flipud(np.transpose(np.vstack([fight.fitx+1, ploty])))])
        right_line_line_pts = np.hstack((right_line_line1, right_line_line2))
      
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
      
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([left_line_line_pts]), (255, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_line_pts]), (255, 255, 0))
        result = cv2.addWeighted(result, 1, window_img, 1.0, 0)
          
        plt.imshow(result)
        plt.plot(left.fitx, ploty, color='yellow')
        plt.plot(fight.fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return ploty, left, right

def draw_lane(dst, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(dst)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp
    
def draw_ctr_line(dst, ploty, left_fitx, right_fitx):
    # Create an image to draw the line on
    warp_zero = np.zeros_like(dst)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Draw center line onto the warped image
    lane_ctr_line = left_fitx + (right_fitx - left_fitx)/2
    ctr_line_line1 = np.array([np.transpose(np.vstack([lane_ctr_line-2, ploty]))])
    ctr_line_line2 = np.array([np.flipud(np.transpose(np.vstack([lane_ctr_line+2, ploty])))])
    lane_ctr_line_pts = np.hstack((ctr_line_line1, ctr_line_line2))
    cv2.fillPoly(color_warp, np.int_([lane_ctr_line_pts]), (75,25,255))

    return color_warp
    
# camera calibration
mtx, dist = camcal('./camera_cal/calibration*.jpg')
#checkcal('./camera_cal/calibration2.jpg', mtx, dist)

# one-time calculation of warp transform matrices
img_size = (1280,720)
M, Minv = warp_transforms(img_size)

frame_num = 0
def process_image(img):
    global frame_num
    
    if img.shape[2] == 4:
        img = img[:,:,0:3]

    dst = np.copy(img)

# #   Save each frame of video
#     fn = './project_video/project_video' + str(frame_num) + '.jpg'
#     mpimg.imsave(fn, img)
    
    # distortion correction
    dst = cv2.undistort(dst, mtx, dist, None, mtx)

    # apply color and gradient threshold pipeline to image
    dst = pipeline(dst)

    # extract only the region of interest in front of the vehicle
    dst = region_of_interest(dst)
        
    # warp the image to a topdown view
    dst = warp(dst, M)
  
    ploty, leftLine, rightLine = fit_lines(dst)
    
    # Draw lane image and warp it back to original image space using inverse perspective matrix (Minv)
    lane = draw_lane(dst, ploty, leftLine.fitx, rightLine.fitx)
    lane_undist = warp(lane, Minv)

    # Combine lane image with the original image
    dst = cv2.addWeighted(img, 1, lane_undist, 0.3, 0)

#     # Draw center line, warp it back to original image space using inverse perspective matrix (Minv), and combine
#     ctr_line = draw_ctr_line(dst, ploty, left_fitx, right_fitx)
#     ctr_line_undist = warp(ctr_line, Minv)
#     dst = cv2.addWeighted(dst, 1, ctr_line_undist, 1.0, 0)

#    dst = np.dstack((dst, dst, dst))*255

    # Write radii as text overlayed on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    rad = (leftLine.radius_of_curvature + rightLine.radius_of_curvature)/2
    msg = 'radius: ' + str(int(rad)) + 'm'
    cv2.putText(dst,msg,(550,540), font, 0.85,(0,0,255),2)
    
    # Write deviation from center as text overlayed on image
    xm_per_pix = 3.7/700
    deviation = 640. - (left.fitx[710] + (right.fitx[710] - left.fitx[710])/2)
    deviation = deviation * xm_per_pix
    msg = 'deviation: ' + "{:2.2f}".format(deviation) + 'm'
    cv2.putText(dst,msg,(520,680), font, 0.85,(0,0,255),2)

    msg = 'frame: ' + str(int(frame_num))
    cv2.putText(dst,msg,(50,50), font, 0.85,(0,0,255),2)
    
    # Draw car's center mark at bottom of screen
    marker = np.array([ [632,719], [640,700], [648,719] ], np.int32)
    cv2.fillPoly(dst, [marker], (0,0,0))

    frame_num += 1
            
    return dst

# # process test images
# imagesPath = './test_images/test1*.jpg'
# #imagesPath = './test_images/project_video*.jpg'
# test_images = glob.glob(imagesPath)
# for filename in test_images:
#     src = mpimg.imread(filename)
#     dst = process_image(src)
#     inspect1(src, dst)

# process the video
clip = VideoFileClip("project_video.mp4")
processed_video = 'project_video_output.mp4'
#clip = VideoFileClip("challenge_video.mp4")
#processed_video = 'challenge_video_output.mp4'
processed_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
processed_clip.write_videofile(processed_video, audio=False)

exit()