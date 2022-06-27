import cv2
from PIL import Image as I
import numpy as np

class Webcam:

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def get_frame(self):

        if self.video.isOpened():    
            rval, frame_raw = self.video.read()

        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        pil_img = I.fromarray(frame.astype('uint8'), "RGB")

        return frame_raw, pil_img
