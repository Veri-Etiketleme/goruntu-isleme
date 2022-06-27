import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image as I
import time
import torch.nn as nn

class MaskConvNet(nn.Module):
    def __init__(self):
      super(MaskConvNet, self).__init__()

      # input dim (3, 100, 100)
      self.conv_block1 = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(2, 2), stride=(1, 1)),
          #nn.BatchNorm2d(num_features=6),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
      )

      # dim (6, 49, 49)
      self.conv_block2 = nn.Sequential(
          nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), stride=(1, 1)),
          #nn.BatchNorm2d(num_features=12),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
      )



      # dim (12, 23 23)
      self.FC_block1 = nn.Sequential(
          nn.Linear(in_features=12*23*23, out_features=50),
          #nn.BatchNorm1d(num_features=50),
          nn.ReLU(),
          nn.Dropout(p=0.5)
      )


      # dim (1, 50)
      self.softmax_block = nn.Sequential(
          nn.Linear(in_features=50, out_features=2)
      )



    def forward(self, x):
      x = self.conv_block1(x)
      x = self.conv_block2(x)
      x = x.view(-1, 12*23*23)
      x = self.FC_block1(x)
      x = self.softmax_block(x)
      return x



class Mask_Detector:

    def __init__(self, model_path):
        """
        Constructor
        path to pretrained CNN model is passed as arguement
        """
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
        self.data_transforms = transforms.Compose([
                            transforms.Resize((100, 100)),
                            transforms.ToTensor(),
                            transforms.Normalize(self.mean, self.std)
                            ])

        self.mask_model = MaskConvNet()
        self.mask_model = torch.load(model_path, map_location=torch.device('cpu'))
        self.classifier = {0:"with mask", 1:"without mask"}


    def classify(self, image):
        """
        Classifier method
        PIL image is to be passed as an argument
        """
        image = self.data_transforms(image)
        self.mask_model.eval() # to evaluate mode

        startTime = time.time()
        outputs = self.mask_model(image.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        conclusion = self.classifier[predicted[0].item()]
        duration_to_predict = time.time() - startTime

        # return (conclusion, duration_to_predict)
        return predicted[0].item()
