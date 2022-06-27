import cv2
import numpy as np

from detect_image import detImage

def detecVideo(videoPath):

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()

    cap = cv2.VideoCapture(videoPath)

    while True:
        _, img = cap.read()

        detImage(img)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
