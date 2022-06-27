import torch
import torchvision
import numpy as np
from PIL import Image as I

class Human_Detector:

    def __init__(self, score_threshold = 0.90):

        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        self.model.eval()

    def detect(self, pil_img):

        image = np.transpose(pil_img, (2,0,1))
        image = image/255
        image_tensor = torch.Tensor(image)
        
        predictions = self.model([image_tensor])
        
        boxes = predictions[0]["boxes"].tolist()
        labels = predictions[0]["labels"].tolist()
        scores = predictions[0]["scores"].tolist() 

        preds = list(zip(boxes, labels, scores))
        people = list(filter( lambda x: (x[1] == 1 and x[2] > 0.90), preds))

        list_of_images = []
        list_of_coords = []

        for i in people:
            list_of_images.append(pil_img.crop(tuple(i[0])))
            list_of_coords.append(tuple(i[0]))

        return list_of_coords ,list_of_images    

