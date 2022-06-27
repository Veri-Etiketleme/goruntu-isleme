import numpy as np
import cv2
import imutils
import pytesseract
import re
import csv
import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image file (here we chose this image, you can change the source to whatever you want)
image = cv2.imread('car_image_5.jpg')

# Resize the image - change width to 500
resized_image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("1- Original Image", resized_image)
image = resized_image

# RGB to Gray scale conversion
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2- Grayscale Conversion", gray_image)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray_bilateral = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("3- Bilateral Filter", gray_bilateral)

# Find Edges of the grayscale image
edged = cv2.Canny(gray_bilateral, 170, 200)
cv2.imshow("4- Canny Edges", edged)

# Find contours based on Edges
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# sort contours based on their area keeping minimum required area as '30' 
# (anything smaller than this will not be considered)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None # we currently have no number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx # This is our approx Number Plate Contour
            break


# Drawing the selected contour on the original image
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)


cv2.imshow("5- Final Image With Number Plate Detected", image)
cv2.imwrite("car_plate.jpg", image)

# Masking the part other than the number plate (by applying AND With Black to non-plate pixles)
mask = np.zeros(gray_image.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)

cv2.imshow('6- Car Plate after removing the other parts of current image', new_image)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))

(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray_image[topx:bottomx-1, topy:bottomy-1]
Cropped = cv2.resize(Cropped,(500,100))

cv2.imshow('7- resized cropped plate',Cropped)

gray_image = cv2.resize( Cropped, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(gray_image, (5,5), 0)
gray_image = cv2.medianBlur(gray_image, 3)

# perform OTSU thresh (using binary inverse since opencv contours work better with white text)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imshow("8- Otsu", thresh)
cv2.waitKey(0)
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# apply dilation 
dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
cv2.imshow("9- Dilation", dilation)

# Now we are going to detect the plate number twice, first with tesseract and then manually.
tesseract_detected_text = pytesseract.image_to_string(Cropped)
print("The detected license plate number by tesseract is:",tesseract_detected_text)

# Now we are going to get the plate text character by character. seems easy right! :)
# find contours
try:
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
except:
    ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# create a copy of gray image
im2 = gray_image.copy()

manually_detected_text = ""
# loop through contours and find letters in license plate (note: all the value numbers bellow are experimental)
for cnt in sorted_contours:

    x,y,w,h = cv2.boundingRect(cnt)
    height, width = im2.shape
    
    # if height of box is not a quarter of total height then skip
    if height / float(h) > 6: continue
    ratio = h / float(w)
    print("height to width ratio:", h / float(w))
    
    # if height to width ratio is less than 1.5 skip
    if (ratio < 0.9) or (ratio > 5): continue
    area = h * w
    
    # if width is not more than 25 pixels skip
    if width / float(w) > 15: continue
    # if area is less than 100 pixels skip
    if area < 100: continue

    # draw the rectangle
    rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
    roi = thresh[y-5:y+h+5, x-5:x+w+5]
    roi = cv2.bitwise_not(roi)
    roi = cv2.medianBlur(roi, 5)

    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    # remove unnecessary symbols and unicodes from the detected character using regex
    text = re.sub('[^A-Za-z0-9]+|\t', '', text)
    manually_detected_text += text


print("The manually detected text is:", manually_detected_text)

cv2.imshow("10- Character's Segmented", im2)

# Now we are going to store the plate number detection details in a CSV file
with open('cars_data.csv', mode='r+', newline='') as csv_file:
  csv_writer = csv.writer(csv_file)
  reader = csv.reader(csv_file)
  lines = len(list(reader))
  if lines < 1:
    csv_writer.writerow(['Datetime', 'Tesseract Result', 'Manual Result'])
  csv_writer.writerow([datetime.datetime.now(), tesseract_detected_text, manually_detected_text])

print("The plate number details have been added to the CSV file.")

cv2.waitKey(0)
cv2.destroyAllWindows()
