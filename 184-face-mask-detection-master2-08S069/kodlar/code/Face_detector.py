import numpy as np
import cv2

class Face_detector():

    def __init__(self):

        self.detector =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    def detect(self, frame):    

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 2)

        return faces