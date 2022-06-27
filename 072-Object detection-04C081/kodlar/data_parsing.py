import cv2
import xml.etree.ElementTree as ET

from numpy.lib.polynomial import roots
import random

i = random.randint(0, 852)


tree = ET.parse("./dataset/annotations/maksssksksss" + str(i) + ".xml")
root = tree.getroot()
print(root.tag)

folder = root[0].text
filename = root[1].text

width = root[2][0].text
height = root[2][1].text

print(folder)
print(filename)
print(width)
print(height)

img = cv2.imread("./dataset/images/" + filename)

for i in range(4, len(root)):
    # print(root[i].tag)
    label = root[i][0].text
    x1 = int(root[i][5][0].text)
    y1 = int(root[i][5][1].text)
    x2 = int(root[i][5][2].text) 
    y2 = int(root[i][5][3].text)
    # print(x1, y1, x2, y2)
    if label == 'without_mask':
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), thickness=5)
    else:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), thickness=5)


# cv2.rectangle(img, (1,1), (50,70), (255,0,0))

cv2.imshow('Image', img)
cv2.waitKey(0)
