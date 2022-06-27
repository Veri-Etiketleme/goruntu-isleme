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
'''
# Non-trainalbe convolutional part for the classification layer of VGG16
class VGG_convNxN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(VGG_convNxN, self).__init__()
        # NxN convolutional layer
        self.conv_NxN = F.Conv2d()
'''
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
        # output_shape = stride * (input_shape - 1) + kernel_size - \
        #2 * padding
        self.upsample_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                    kernel_size=kernel_size_up,
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
classifier_layers = namedtuple('classifier_layers', ['linear_0', 'relu_0',
                        'dropout_0', 'linear_1', 'relu_1', 'dropout_1'])
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
        self.vgg_copy_features = copy.deepcopy(
            vgg_pretrained_model.features)
        self.vgg_copy_classifier = copy.deepcopy(
            vgg_pretrained_model.classifier)

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
                weight = module.weight.view(module.weight.size(0),
                                            x.size(1), 7, 7)
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
                weight = module.weight.view(module.weight.size(0),
                                            x.size(1), 1, 1)
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

        return feature_layers(**output_features), classifier_layers(
            **output_classifier)

class FCN(nn.Module):
    def __init__(self, vIn_channels, num_classes, vKernel_size_up,
                    vStride_up, vPadding_up):
        super(FCN, self).__init__()
        # the v in object initializing variables determines that they are
        # vectors
        # the fist element of the vector is for dropout_1, maxpool_layer_4,
        # maxpool_layer_3
        # create the object of 1x1 convolutional module for dropout_1
        self.conv1x1_model_7 = FCN_conv1x1(vIn_channels[0], num_classes)
        # create the object of 1x1 convolutional module for maxpool_4
        self.conv1x1_model_4 = FCN_conv1x1(vIn_channels[1], num_classes)
        # create the object of 1x1 convolutional module for maxpool_3
        self.conv1x1_model_3 = FCN_conv1x1(vIn_channels[2], num_classes)
        # upsampling of dropout_1 layer
        self.up_model_7 = FCN_upsample(num_classes, num_classes,
                        vKernel_size_up[0], vStride_up[0], vPadding_up[0])
        # upsample the dropout_1 once again
        #self.up_model_7_1 = FCN_upsample(num_classes, num_classes,
        # vKernel_size_up[1], vStride_up[1], vPadding_up[1])
        # upsampling of maxpool_4
        self.up_model_4 = FCN_upsample(num_classes, num_classes,
                        vKernel_size_up[1], vStride_up[1], vPadding_up[1])
        # upsampling of maxpool_3
        self.up_model_3 = FCN_upsample(num_classes, num_classes,
                        vKernel_size_up[2], vStride_up[2], vPadding_up[2])
        #self.classify = FCN_conv1x1(num_classes, 3)
        # The final dense classification of the upsampled image
        #self.classifier = nn.Sigmoid()

    def forward(self, x_f, x_cl):
        # get the maxpooled outputs from maxpool_7, maxpool_4 and maxpool_3
        # of vgg16 model
        # x = self.vgg_feature_model(x)
        # upsample the maxpool_7 so that it is compatible with maxpool_4
        # for matrix addition
        # x[0] corresponds to the maxpool_3 of vgg16 features
        # (check class vgg_features) #
        # x[1] corresponds to the maxpool_4 of vgg16 features
        # (check class vgg_features) #
        # x[2] corresponds to the dropout_1 of vgg16 features
        # (check class vgg_features) #
        x7_1x1 = self.conv1x1_model_7(x_cl[5]) # 1x1 convolve the maxpool_7
        x4_1x1 = self.conv1x1_model_4(x_f[1]) # 1x1 convolve the maxpool_4
        x3_1x1 = self.conv1x1_model_3(x_f[0]) # 1x1 convolve the maxpool_3
        # upsample the 1x1 convolved dropout_1
        x7_upsample = self.up_model_7(x7_1x1)
        #print(x7_upsample.size())
        #print(x4_1x1.size())
        #print(x7_1x1.size())
        # add upsampled dropout_1 and 1x1 convolved maxpool_4 and generate
        # the skipped connection
        x7x4_skip = torch.add(x7_upsample, x4_1x1)
        # upsample the x7x4_skip connection
        #print(x7x4_skip.size())
        x7x4_upsample = self.up_model_4(x7x4_skip)
        #print(x7x4_upsample.size())
        #print(x3_1x1.size())
        # add upsampled x7x4_skip and 1x1 convolved maxpool_3 and generate
        # the second skipped connection
        x7x4x3_skip = torch.add(x7x4_upsample, x3_1x1)
        #print(x3_1x1.size())
        #print(x7x4_upsample.size())
        # Finally upsample the skipped connection so that it has the shape
        # of input image and return it
        y = self.up_model_3(x7x4x3_skip)
        #y = torch.tanh(y)
        #print(x7x4x3_skip.size())
        #y = self.classifier_layer(y)
        #y = self.classifier(y)
        return y
'''
# dummy image loader test
dummy_img = image_loader('./data/dummy_abba.jpg', DEVICE)
dummy_img_cpu = dummy_img.cpu()
dummy_img_cpu = dummy_img_cpu.squeeze(0)
dummy_img_cpu = np.swapaxes(dummy_img_cpu, 0, 1)
dummy_img_cpu = np.swapaxes(dummy_img_cpu, 1, 2)
print(dummy_img_cpu.shape)
plt.imshow(dummy_img_cpu.numpy())
plt.show()

# dummy FCN model variables
d_vInChannels = [512, 512, 256]
d_numClasses = 10
d_vKernelSize = [4, 4, 4]
d_vStrides = [2, 2, 2]
d_vPadding = [1, 1, 1]

# dummy feature extractor
vgg_dummy = vgg.vgg16(pretrained=True).features
feature_model = vgg_features(vgg_dummy).to(DEVICE).eval()

# create the vgg feature model object
vgg_feature_model = vgg_features(vgg_dummy).to(DEVICE).eval()
# get the maxpooled features
d_vgg_out = vgg_feature_model(dummy_img)
print(d_vgg_out[1].size())
# dummy FCN
dummy_FCN = FCN(d_vInChannels, d_numClasses, d_vKernelSize, d_vStrides, d_vPadding).to(DEVICE)
d_fcn_out = dummy_FCN(d_vgg_out)

# show the fcn output
d_fcn_out_cpu = d_fcn_out.cpu()
d_fcn_out_cpu = d_fcn_out_cpu.detach().numpy()
d_fcn_out_cpu = d_fcn_out_cpu.squeeze(0)
d_fcn_out_cpu = np.swapaxes(d_fcn_out_cpu, 0, 1)
d_fcn_out_cpu = np.swapaxes(d_fcn_out_cpu, 1, 2)
d_fcn_out_cpu = np.mean(d_fcn_out_cpu, axis=2)
print(d_fcn_out_cpu.shape)
plt.imshow(d_fcn_out_cpu)
plt.show()
'''
'''
# dummy image loader test
dummy_img = image_loader('./data/dummy_abba.jpg', DEVICE)
dummy_img_cpu = dummy_img.cpu()
dummy_img_cpu = dummy_img_cpu.squeeze(0)
dummy_img_cpu = np.swapaxes(dummy_img_cpu, 0, 1)
dummy_img_cpu = np.swapaxes(dummy_img_cpu, 1, 2)
print(dummy_img_cpu.shape)
plt.imshow(dummy_img_cpu.numpy())
plt.show()
# dummy feature extractor
vgg_dummy = vgg.vgg16(pretrained=True)
dummy_vgg_model = vgg_features(vgg_dummy).to(DEVICE).eval()
# dummy vgg16 output
vgg_out_features, vgg_out_cl = dummy_vgg_model(dummy_img)
#print(vgg_out_features[1].size())
#print(vgg_out_cl[5].size())
# dummy 1x1 convolve the vgg_out_cl[5]
d_cl_conv1x1_layer = FCN_conv1x1(4096, 10).to(DEVICE)
cl_1x1 = d_cl_conv1x1_layer(vgg_out_cl[5])
# dummy upsample the vgg_out_cl[5]
d_cl_upsample_layer = FCN_upsample(10, 10, 4, 2, 1).to(DEVICE)
d_cl_upsample = d_cl_upsample_layer(cl_1x1)
#print(d_cl_upsample.size())
'''
'''
vgg_out_cpu = vgg_out_cl[5].cpu()
print(vgg_out_cpu.shape)
vgg_out_cpu = vgg_out_cpu.detach().numpy()
vgg_out_cpu = vgg_out_cpu.squeeze(0)
vgg_out_cpu = np.swapaxes(vgg_out_cpu, 0, 1)
vgg_out_cpu = np.swapaxes(vgg_out_cpu, 1, 2)
vgg_out_cpu = np.mean(vgg_out_cpu, axis=2)
#print(vgg_out_cpu.shape)
plt.imshow(vgg_out_cpu)
plt.show()

# dummy upsample test
dummy_upsample_module = FCN_upsample(256, 256, 4, 2, 1).to(DEVICE)
dummy_upsample = dummy_upsample_module(vgg_out[2])
dummy_upsample_cpu = dummy_upsample.cpu()
dummy_upsample_cpu = dummy_upsample_cpu.detach().numpy()
dummy_upsample_cpu = dummy_upsample_cpu.squeeze(0)
dummy_upsample_cpu = np.swapaxes(dummy_upsample_cpu, 0, 1)
dummy_upsample_cpu = np.swapaxes(dummy_upsample_cpu, 1, 2)
dummy_upsample_cpu = np.mean(dummy_upsample_cpu, axis=2)
#print(dummy_upsample.shape)
plt.imshow(dummy_upsample_cpu)
plt.show()
'''
