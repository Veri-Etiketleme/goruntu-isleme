import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from pytorch_FCN_v2 import FCN, vgg_features
import vgg

from utils import *
import matplotlib.pyplot as plt

# check for available DEVICEs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
# Global variables
IN_CHANNELS = [4096, 512, 256] # vgg input channels
NUM_CLASSES = 2 # Number of classes
KERNEL_SIZE = [4, 4, 16] # Upsample kernel sizes
STRIDES = [2, 2, 8] # Upsample strides
PADDING = [1, 1, 4] # Upsample padding

NUM_EPOCHS = 200 # Number of training epochs
BATCH_SIZE = 20 # Training batch size
NUM_BATCHES = 1 # Number of training batches
LR = 1e-2 # 0.000000001 # Learing rate
# Folder containing training images
IMAGE_FOLDER = './data/ICDAR2017_simple/input_target_data'
# Folder containing training images
#IMAGE_FOLDER_TEST = './data/VOCdevkit/VOC2012'
# folder save test input images
#IMAGE_SAVE_PATH = './data/VOCdevkit/VOC2012/input_images/'
# folder to save predictions
#PREDICTION_SAVE_PATH = './data/VOCdevkit/VOC2012/predicted_images/'
IMAGE_SHAPE = [160, 576] # Image sizes for random cropping
 # Model save path and name
MODEL_NAME_PATH = './model_dict/fcn_v2_ICDAR2017.pt'

def cross_entropy2d(inputs, targets, num_classes, size_average=True):
    n, c, h, w = inputs.size()
    # log_p : (n*h*w, c)
    log_p = inputs.transpose(1, 2).transpose(2, 3).contiguous()
    #print(log_p)
    log_p = F.log_softmax(log_p, dim=1)
    #log_p = log_p[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    log_p = log_p.repeat(targets.size(1), 1)
    #print(torch.min(log_p))
    # targets: (n*h*w, )
    mask = targets >= 0
    targets = targets[mask]
    #print(torch.max(targets))
    loss = F.nll_loss(log_p, targets, size_average=False)
    #print(loss)

    if size_average:
        mask = mask.type(torch.float)
        #loss.type(torch.float)
        loss /= torch.sum(mask.data)

    return loss

def soft_dice_loss(inputs, targets, epsilon=1e-6):
    numerator = 2 * torch.sum(inputs * targets)
    denominator = torch.sum(inputs**2 + targets**2)

    return 1 - torch.mean(numerator / (denominator + epsilon))

def optimization(model, learning_rate):
    # define the loss function
    #critarion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate,
    #                      momentum=0.9)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
    #                          momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                          gamma=0.5)

    return optimizer, scheduler

def train(image_folder, batch_size):
    critarion = nn.MSELoss()
    # setup the batch generator
    get_batch = data_generator(IMAGE_FOLDER, IMAGE_SHAPE,
                              DEVICE)
    # make the FCN model
    model = FCN(IN_CHANNELS, NUM_CLASSES, KERNEL_SIZE, STRIDES,
                PADDING).to(DEVICE).train()
    # get the optimization related objects
    optimizer, scheduler = optimization(model, LR)
    # make the vgg16 feature model
    vgg_model = vgg.vgg16(pretrained=True)
    vgg_layers = vgg_features(vgg_model).to(DEVICE).eval()
    # freeze the vgg layers
    for param in vgg_layers.parameters():
        param.requires_grad = False

    for epoch in range(NUM_EPOCHS):
        # update the scheduler
        scheduler.step()
        # list to hold the batch loss to compute the final epoch loss for
        # each epoch
        batch_loss = []
        for image, label in get_batch(BATCH_SIZE, NUM_BATCHES):
            #print(torch.max(label))
            #print(image.size())
            #print(label.size())
            optimizer.zero_grad() # set all gradients to zeor
            #print(torch.max(image))
             # get input image features from vgg16
            x_feature, x_classes = vgg_layers(image)
            # get the FCN predictions
            prediction = model(x_feature, x_classes)
            #print(prediction.size())
            #print(label.size())
            # reshape labels to calculate the loss
            #label = label.view(-1, NUM_CLASSES)
            # reshape predictions to calculate the loss
            #prediction = prediction.view(-1, NUM_CLASSES)
            #loss = cross_entropy2d(prediction, label, NUM_CLASSES)
            loss = soft_dice_loss(prediction, label)
            loss.backward()
            optimizer.step()
            #quick_view(prediction)
            # print batch loss
            #print('Epoch {} of {}'.format(epoch+1, NUM_EPOCHS),
            #      'Training loss {:.5f}'.format(loss.item()))
            batch_loss.append(loss.item())
        print('Epoch loss {:.5f}'.format(np.mean(np.array(batch_loss))),
              'at Epoch {}'.format(epoch+1))
    # Save the model
    model_param = model.state_dict() # get the trained model parameters
    torch.save(model_param, MODEL_NAME_PATH)
    print('model saved at: {}'.format(MODEL_NAME_PATH))

def test(model_name_path):
    # Set the trained model and vgg layers in evaluation mode
    model = FCN(IN_CHANNELS, NUM_CLASSES, KERNEL_SIZE, STRIDES,
                PADDING).to(DEVICE).eval()
    # vgg model initiation
    vgg_model = vgg.vgg16(pretrained=True)
    vgg_layers = vgg_features(vgg_model).to(DEVICE).eval()
    # Load the trained params
    model_param = torch.load(model_name_path)
    # Update the model state dictionary with loaded params
    model.load_state_dict(model_param)
    # Initiate the test generator
    get_test_image = data_generator(IMAGE_FOLDER,
                                   IMAGE_SHAPE, DEVICE)
    count = 0
    for image, _ in get_test_image(BATCH_SIZE, NUM_BATCHES):
        count += 1
        #print(image.size())
        # get the vgg features and dense classification
        tx_feature, tx_classes = vgg_layers(image)
        tprediction = model(tx_feature, tx_classes)
        #print(tprediction.size())
        #quick_view(image, tprediction, NUM_CLASSES)
        save_prediction(tprediction, image, count)

def main():
    #fcn = train(IMAGE_FOLDER, BATCH_SIZE)
    test(MODEL_NAME_PATH)

if __name__ == '__main__':
    main()
