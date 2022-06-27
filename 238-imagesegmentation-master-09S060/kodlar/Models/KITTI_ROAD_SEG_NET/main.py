import torch
import torch.optim as optim
import torch.nn as nn

from pytorch_FCN import FCN, vgg_features
import vgg

from utils import *
import matplotlib.pyplot as plt

# check for available DEVICEs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
IN_CHANNELS = [4096, 512, 256] # vgg input channels
NUM_CLASSES = 2 # Number of classes
KERNEL_SIZE = [4, 4, 16] # Upsample kernel sizes
STRIDES = [2, 2, 8] # Upsample strides
PADDING = [1, 1, 4] # Upsample padding

NUM_EPOCHS = 300 # Number of training epochs
BATCH_SIZE = 8 # Training batch size
LR = 1e-3 # Learing rate

IMAGE_FOLDER = './data/data_road/training/' # Folder containing training images
IMAGE_FOLDER_TEST = './data/data_road/testing/' # Folder containing training images
IMAGE_SAVE_PATH = './data/data_road/testing/input_images/'
PREDICTION_SAVE_PATH = './data/data_road/testing/predicted_images/'
IMAGE_SHAPE = [576, 160] # Image sizes for resizing

MODEL_NAME_PATH = './model_dict/fcn_v2.pt' # Model save path and name

def optimization(model, learning_rate):
    # define the loss function
    critarion = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    return critarion, optimizer, scheduler

def train(image_folder, batch_size):
    # setup the batch generator
    get_batch = data_generator(IMAGE_FOLDER, IMAGE_SHAPE, DEVICE)
    # make the FCN model
    model = FCN(IN_CHANNELS, NUM_CLASSES, KERNEL_SIZE, STRIDES, PADDING).to(DEVICE).train()
    # get the optimization related objects
    loss_fn, optimizer, scheduler = optimization(model, LR)
    # make the vgg16 feature model
    vgg_model = vgg.vgg16(pretrained=True)
    vgg_layers = vgg_features(vgg_model).to(DEVICE).eval()
    # freeze the vgg layers
    for param in vgg_layers.parameters():
        param.requires_grad = False

    for epoch in range(NUM_EPOCHS):
        # update the scheduler
        scheduler.step()
        # list to hold the batch loss to compute the final epoch loss for each epoch
        batch_loss = []
        for image, label in get_batch(BATCH_SIZE):
            #print(label.size())
            optimizer.zero_grad() # set all gradients to zeor
            #print(image.size())
            x_feature, x_classes = vgg_layers(image) # get input image features from vgg16
            prediction = model(x_feature, x_classes) # get the FCN predictions
            label = label.view(-1, NUM_CLASSES) # reshape labels to calculate the loss
            prediction = prediction.view(-1, NUM_CLASSES) # reshape predictions to calculate the loss
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()
            # print batch loss
            print('Epoch {} of {}'.format(epoch+1, NUM_EPOCHS), 'Training loss {:.5f}'.format(loss.item()))
            batch_loss.append(loss.item())
        print('Epoch loss {:.5f}'.format(np.mean(np.array(batch_loss))), 'at Epoch {}'.format(epoch+1))
    # Save the model
    model_param = model.state_dict() # get the trained model parameters
    torch.save(model_param, MODEL_NAME_PATH)
    print('model saved at: {}'.format(MODEL_NAME_PATH))

def test(model_name_path):
    # Set the trained model and vgg layers in evaluation mode
    model = FCN(IN_CHANNELS, NUM_CLASSES, KERNEL_SIZE, STRIDES, PADDING).to(DEVICE).eval()
    # vgg model initiation
    vgg_model = vgg.vgg16(pretrained=True)
    vgg_layers = vgg_features(vgg_model).to(DEVICE).eval()
    # Load the trained params
    model_param = torch.load(model_name_path)
    # Update the model state dictionary with loaded params
    model.load_state_dict(model_param)
    # Initiate the test generator
    get_test_image = test_generator(IMAGE_FOLDER_TEST, IMAGE_SHAPE, DEVICE)
    for image, image_idx in get_test_image():
        # get the vgg features and dense classification
        tx_feature, tx_classes = vgg_layers(image)
        tprediction = model(tx_feature, tx_classes)
        save_prediction(image, tprediction, image_idx, IMAGE_SAVE_PATH, PREDICTION_SAVE_PATH)

def main():
    fcn = train(IMAGE_FOLDER, BATCH_SIZE)
    test(MODEL_NAME_PATH)

if __name__ == '__main__':
    main()
