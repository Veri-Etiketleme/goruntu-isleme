import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from collections import namedtuple
import copy

import vgg

from utils import *

# check for available DEVICEs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# FCN trainable part implementation : 1x1 convolution and Upsampling modules
class FCN_conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN_conv1x1, self).__init__()
        # 1x1 convolution of the maxpooled features from vgg net
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    stride=1, padding=0)
        #self.relu_1x1 = nn.ReLU(inplace=True)
        # initialize the conv_1x1 weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv_1x1.weight, gain=1.4)

    def forward(self, x):
        y = self.conv_1x1(x)
        return y

class FCN_upsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size_up, stride_up, padding_up):
        super(FCN_upsample, self).__init__()
        # Create the upsampling layers
        # transpose convolution arithmatics : with_padding, non-unity_stride
        # output_shape = stride * (input_shape - 1) + kernel_size - 2 * padding
        self.upsample_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size_up,
                                    stride=stride_up, padding=padding_up)
        #self.relu_upsample = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.upsample_layer.weight, gain=1.4)

    def forward(self, x):
        y = self.upsample_layer(x) # upsample the convolved features
        return y

# Define the feature extractor. In this case a vgg16
# The vgg16 layers we need to extract the features
feature_layers = namedtuple('feature_layers', ['maxpool_3', 'maxpool_4'])
classifier_layers = namedtuple('classifier_layers', ['linear_0', 'relu_0', 'dropout_0',
                                'linear_1', 'relu_1', 'dropout_1'])
class vgg_features(nn.Module):
    def __init__(self, vgg_pretrained_model):
        super(vgg_features, self).__init__()
        self.feature_name_mapping = {
            '16': "maxpool_3",
            '23': "maxpool_4"
        }
        self.classifier_name_mapping = {
            '0': "linear_0",
            '1': "relu_0",
            '2': "dropout_0",
            '3': "linear_1",
            '4': "relu_1",
            '5': "dropout_1"
        }
        # define convolutional layers for the classification part
        self.fc_6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.fc_7 = nn.Conv2d(4096, 4096, 1)
        # copy the vgg model
        self.vgg_copy_features = copy.deepcopy(vgg_pretrained_model.features)
        self.vgg_copy_classifier = copy.deepcopy(vgg_pretrained_model.classifier)

    def forward(self, x):
        # feature dependent part
        output_features = {}
        i = 0
        # extract the features from given Layers
        for module in self.vgg_copy_features.children():
            #print(module)
            #print(x.size())
            x = module(x)
            if str(i) in self.feature_name_mapping:
                #print(module)
                #print(x.size())
                output_features[self.feature_name_mapping[str(i)]] = x
            i += 1
        # classifier dependent part
        #print(x.size())
        output_classifier = {}
        j = 0
        # extract the classifier outputs
        for module in self.vgg_copy_classifier.children():
            if j == 0:
                #print(module)
                #x = F.pad(x, (6, 6, 1, 1), mode='constant', value=0)
                #print(x.size())
                weight = module.weight.view(module.weight.size(0), x.size(1), 7, 7)
                #print(weight.size())
                bias = module.bias
                #print(bias.size())
                #print(x.size())
                self.fc_6.weight.data.copy_(weight)
                self.fc_6.bias.data.copy_(bias)
                if torch.mean(self.fc_6.weight.data) != torch.mean(weight):
                    print('Weight initialization failed')
                if torch.mean(self.fc_6.bias.data) != torch.mean(bias):
                    print('bias initialization failed')
                x = self.fc_6(x)
                #x = F.conv2d(x, weight, bias)
                #print(x.size())
            if j == 3:
                weight = module.weight.view(module.weight.size(0), x.size(1), 1, 1)
                bias = module.bias
                self.fc_7.weight.data.copy_(weight)
                self.fc_7.bias.data.copy_(bias)
                if torch.mean(self.fc_7.weight.data) != torch.mean(weight):
                    print('Weight initialization failed')
                if torch.mean(self.fc_7.bias.data) != torch.mean(bias):
                    print('bias initialization failed')
                x = self.fc_7(x)
                #x = F.conv2d(x, weight, bias)
                #print(x.size())
                #print(module.weight.size())
            if j != 0 and j != 3 and j != 6:
                #print(module)
                x = module(x)
                #print(x.size())
            if str(j) in self.classifier_name_mapping:
                #print(module)
                output_classifier[self.classifier_name_mapping[str(j)]] = x
            j += 1
        #print(output)

        return feature_layers(**output_features), classifier_layers(**output_classifier)

class FCN(nn.Module):
    def __init__(self, vIn_channels, num_classes, vKernel_size_up,
                    vStride_up, vPadding_up):
        super(FCN, self).__init__()
        # the v in object initializing variables determines that they are vectors
        # the fist element of the vector is for dropout_1, maxpool_layer_4, maxpool_layer_3
        # create the object of 1x1 convolutional module for dropout_1
        self.conv1x1_model_7 = FCN_conv1x1(vIn_channels[0], num_classes)
        # create the object of 1x1 convolutional module for maxpool_4
        self.conv1x1_model_4 = FCN_conv1x1(vIn_channels[1], num_classes)
        # create the object of 1x1 convolutional module for maxpool_3
        self.conv1x1_model_3 = FCN_conv1x1(vIn_channels[2], num_classes)
        # upsampling of dropout_1 layer
        self.up_model_7 = FCN_upsample(num_classes, num_classes, vKernel_size_up[0],
                                        vStride_up[0], vPadding_up[0])
        # upsample the dropout_1 once again
        #self.up_model_7_1 = FCN_upsample(num_classes, num_classes, vKernel_size_up[1],
        #                                vStride_up[1], vPadding_up[1])
        # upsampling of maxpool_4
        self.up_model_4 = FCN_upsample(num_classes, num_classes, vKernel_size_up[1],
                                        vStride_up[1], vPadding_up[1])
        # upsampling of maxpool_3
        self.up_model_3 = FCN_upsample(num_classes, num_classes, vKernel_size_up[2],
                                        vStride_up[2], vPadding_up[2])
        # The final dense classification of the upsampled image
        #self.classifier = nn.ReLU(inplace=True)

    def forward(self, x_f, x_cl):
        # get the maxpooled outputs from maxpool_7, maxpool_4 and maxpool_3 of vgg16 model
        # x = self.vgg_feature_model(x)
        # upsample the maxpool_7 so that it is compatible with maxpool_4 for matrix addition
        # x[0] corresponds to the maxpool_3 of vgg16 features (check class vgg_features) #
        # x[1] corresponds to the maxpool_4 of vgg16 features (check class vgg_features) #
        # x[2] corresponds to the dropout_1 of vgg16 features (check class vgg_features) #
        x7_1x1 = self.conv1x1_model_7(x_cl[5]) # 1x1 convolve the maxpool_7
        x4_1x1 = self.conv1x1_model_4(x_f[1]) # 1x1 convolve the maxpool_4
        x3_1x1 = self.conv1x1_model_3(x_f[0]) # 1x1 convolve the maxpool_3
        # upsample the 1x1 convolved dropout_1
        x7_upsample = self.up_model_7(x7_1x1)
        #print(x4_1x1.size())
        #print(x7_1x1.size())
        # add upsampled dropout_1 and 1x1 convolved maxpool_4 and generate the skipped connection
        x7x4_skip = torch.add(x7_upsample, x4_1x1)
        # upsample the x7x4_skip connection
        x7x4_upsample = self.up_model_4(x7x4_skip)
        # add upsampled x7x4_skip and 1x1 convolved maxpool_3 and generate the second skipped connection
        x7x4x3_skip = torch.add(x7x4_upsample, x3_1x1)
        #print(x3_1x1.size())
        #print(x7x4_upsample.size())
        # Finally upsample the skipped connection so that it has the shape of input image and return it
        y = self.up_model_3(x7x4x3_skip)
        #print(x7x4x3_skip.size())
        #y = self.classifier_layer(y)
        #y = self.classifier(y)
        return y

