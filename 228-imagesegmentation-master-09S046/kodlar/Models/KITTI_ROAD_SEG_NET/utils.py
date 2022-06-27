from PIL import Image
import prepare_image as pimg
import glob
import os
import re
import PIL
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
import numpy as np
import scipy
import torch

def image_loader(image_name, device_id, im_size=256, resize_img=False, img_type='RGB'):
    image = Image.open(image_name) ## Open the image
    if resize_img:
        new_size = [im_size, im_size]
        image = image.resize(new_size, resample=PIL.Image.BICUBIC)
    #image = pimg.resize_img(image, imsize_tuple)
    image_array = np.array(image)#.transpose(2, 0, 1) ## Make it an np array
    #print(image_array.shape)
    if img_type == 'gray':
        image_array = np.reshape(image_array, [imsize, imsize, 1])
    image_array = image_array / 255.0
    image_array = image_array.transpose(2, 0, 1)
    image_torch = pimg.np_to_torch(image_array) ## Conver it to tensor
    image_torch = image_torch.unsqueeze(0) ## Make a fake batch dimension
    return image_torch.to(device_id, torch.float)

def data_generator(image_folder, image_shape, device):
    # Generate data batches
    def batch_generator(batch_size):
        # get the image names as an array of strings from the data folder
        image_paths = glob.glob(os.path.join(image_folder, 'image_2', '*.png'))
        # get the image labels
        # re.sub(pattern, repl, string) replaces the pattern: _lane|road_ with repl: _
        # for the a given string. the os.path.basename retuns the second element of a string
        label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob.glob(os.path.join(image_folder, 'gt_image_2', '*_road_*.png'))
        } # This creates a label dictionary
        #yield image_paths
        #print(label_paths)
        background_color = np.array([255, 0, 0])
        # shuffle the data
        np.random.shuffle(image_paths)
        # loop for number of batches
        for batch in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            # loop over the file name array and get the corresponding batch
            for image_file in image_paths[batch:batch+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)] # get the gt image filename

                # load the images and gt images
                image = Image.open(image_file)
                image = image.resize(image_shape, PIL.Image.BICUBIC)
                image = np.array(image) / 255.0  # Normalize input image
                #print(image.shape)
                image = image.transpose(2, 0, 1)
                gt_image = Image.open(gt_image_file)
                gt_image = gt_image.resize(image_shape, PIL.Image.BICUBIC)
                gt_image = np.array(gt_image)

                gt_bg = np.all(gt_image == background_color, axis=2) # Convert the image to binary
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1) # extend the channel axis
                # get the binary and binary inverted images together
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                gt_image = gt_image.astype('int')
                gt_image = gt_image.transpose(2, 0, 1)
                #print(gt_image.shape)
                images.append(image)
                gt_images.append(gt_image)

            yield torch.from_numpy(np.array(images)).to(device, torch.float), \
                    torch.from_numpy(np.array(gt_images)).to(device, torch.float)
        #print(np.array(gt_images).shape)

    return batch_generator

def test_generator(image_folder, image_shape, device):
    def image_generator():
        # get the image names as an array of strings from the data folder
        image_paths = glob.glob(os.path.join(image_folder, 'image_2', '*.png'))
        image_count = 0
        for image_file in image_paths: # iterate over the file name vector
            image_count += 1 # count the number of test images

            image = Image.open(image_file) # open the test image
            image = image.resize(image_shape, PIL.Image.BICUBIC) # resize the image
            image = np.array(image) / 255.0  # Normalize input image
            image = image.transpose(2, 0, 1) # transpose image to NCHW
            image = np.expand_dims(image, axis=0) # fake batch size

            # yield the image
            yield torch.from_numpy(image).to(device, torch.float), image_count
    return image_generator

def save_prediction(image, prediction, image_index, save_folder_input, save_folder_prediction):
    #print(image_index)
    # convert input and prediction image into numpy with a shape of HWC
    prediction_np = prediction.cpu().detach().numpy() # to numpy
    np_predicted = np.transpose(prediction_np, (0, 2, 3, 1)).squeeze(0) # HWC conversion
    np_predicted = np_predicted[:, :, 1, np.newaxis] # get the road segement
    np_predicted = np.where(np_predicted > 0.5, 1, 0) # binarize the prediction
    mask = np.dot(np_predicted, np.array([[0, 255, 0, 127]])) # make a mask (for road) from predictions RGBA
    mask = scipy.misc.toimage(mask, mode='RGBA') # convert the ndarray to PIL.Image (using scipy)
    #np_predicted = np_predicted * 255.0 # unnormalize and convert to uint8 image

    np_image = image.cpu().numpy() # to numpy
    np_image = np.transpose(np_image, (0, 2, 3, 1)).squeeze(0) # HWC conversion
    np_image = np.uint8(np_image * 255.0) # unnormalize and convert to uint8 image

    original_image = scipy.misc.toimage(np_image) # convert the input image to PIL image (using scipy)
    original_image.paste(mask, box=None, mask=mask) # paste the mask on top of the original input image
    #print(type(original_image))
    '''
    plt.subplot(121)
    plt.imshow(original_image, cmap=plt.cm.gray)
    plt.subplot(122)
    plt.imshow(np_image)
    plt.show()
    '''
    # save the images using PIL Image
    image = Image.fromarray(np_image).convert('RGB')
    input_image_name = save_folder_input + 'input_image_' + str(image_index) + '.png'
    image.save(input_image_name, 'PNG')

    # save the predictions using PIL Image
    predicted_image_name = save_folder_prediction + 'prediction_' + str(image_index) + '.png'
    original_image.save(predicted_image_name, 'PNG')

def parser_xml(directory, class_name):
    # get the list of xml files
    file_list = glob.glob(os.path.join(directory, '*.xml'))
    # create an empty dict to hold class_name : files pair
    class_dict = dict()
    #print(file_list)
    for i in range(len(class_name)):
        file_names = []
        # get the current class name
        preferred_class = class_name[i]
        #class_dict.update({preferred_class:[]})
        print(preferred_class)
        for j in range(len(file_list)):
            # get the current file name
            current_file = file_list[j]
            # open the file
            xml_tree = ET.parse(current_file)
            # get the xml root
            root = xml_tree.getroot()
            # look for the 'object' element in the root
            for obj in root.iter('object'):
                # find the 'name' sub_element
                for name in obj.iter('name'):
                    present_classes = name.text
                    if present_classes == preferred_class:
                        #print('got it ;)')
                        file_names.append(current_file)
                        #print(file_names)
                        class_dict.update({preferred_class: file_names})
                        #print(class_dict)
                        break
    print(class_dict['car'])

