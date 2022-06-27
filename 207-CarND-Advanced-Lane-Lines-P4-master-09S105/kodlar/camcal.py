import numpy as np
import cv2
import glob

def calibrate(imagesPath, nx=9, ny=6):
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

def checkcal(img, mtx, dist):
    """
    For visual inspection to validate the correctness of camera calibration metrics.
    """    
    src = mpimg.imread(img)
    dst = cv2.undistort(src, mtx, dist, None, mtx)
    inspect1(src,dst)    
