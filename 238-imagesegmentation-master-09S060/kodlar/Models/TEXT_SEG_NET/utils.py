from PIL import Image
import glob
import os
import re
import PIL
import matplotlib.pyplot as plt
import cv2

import xml.etree.ElementTree as ET
import numpy as np
import scipy
import torch

def data_generator(image_folder, image_shape, device):
    # Generate data batches
    # get the image names as an array of strings from the data folder
    file_paths = glob.glob(os.path.join(image_folder, 'xml_info', '*.xml'))
    print(len(file_paths))
    image_shape_x = int(image_shape[1] / 2)
    image_shape_y = int(image_shape[0] / 2)
    def batch_generator(batch_size, num_batches):
        np.random.seed(1234)
        #file_paths_batch = file_paths[batch:batch+batch_size]
        images = []
        # loop over the file name array and get the corresponding image
        for xml_file in file_paths:
            count = 0
            # give the current xml file to xml parser
            image_name, coordi = parser_xml(xml_file)
            image_path = os.path.join(image_folder, 'input_images', image_name)
            # load the image as an np array
            input_image = cv2.imread(image_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            # get the image dimensions
            im_h, im_w = input_image.shape[0], input_image.shape[1]
            # make a black image of the corresponding image
            gt_image = np.ones((im_h, im_w)) * 255.0
            # coordinate holders
            x_list_max = []
            y_list_max = []
            # Draw the textline using the coordinates
            for key, value in coordi.items():
                # get all the coordinates for the x axis and y axis
                x_axis_elements = np.array(value)[:, 0]
                y_axis_elements = np.array(value)[:, 1]
                # get the min and max element for each coordinate sets
                current_x_max = np.max(x_axis_elements)
                current_y_max = np.max(y_axis_elements)
                x_list_max.append(current_x_max)
                y_list_max.append(current_y_max)
                #print(key)
                #print(value)
                #print('')
                cv2.fillPoly(gt_image, pts=np.int32([value]), color=[0, 0, 0])
            #cv2.imwrite(os.path.join(image_folder, 'image_test', image_name), gt_image)
            x_max = np.max(np.array(x_list_max))
            y_max = np.max(np.array(y_list_max))
            #print(y_max)
            #print(input_image.shape[1])
            # use the min max of coordinates and make a uniform distribution for each axis
            x_distrib = list(map(int, np.random.uniform(low=image_shape[1], high=x_max, size=batch_size)))
            y_distrib = list(map(int, np.random.uniform(low=image_shape[0], high=y_max, size=batch_size)))
            x_distrib_max_idx = np.argmax(x_distrib)
            y_distrib_max_idx = np.argmax(y_distrib)
            #print(x_distrib)
            #print(y_distrib)
            if (input_image.shape[1] - x_distrib[x_distrib_max_idx]) < image_shape[1]:
                #print(input_image.shape[1] - x_distrib[x_distrib_max_idx])
                #print('x_max')
                #print(x_distrib_max_idx)
                #print(x_distrib[x_distrib_max_idx])
                #print(input_image.shape[1])
                x_distrib[x_distrib_max_idx] = x_distrib[x_distrib_max_idx] - image_shape[1]
                #print(x_distrib[x_distrib_max_idx])
                #if (input_image.shape[0] - y_distrib[x_distrib_max_idx]) < image_shape[0]:
                #    y_distrib[x_distrib_max_idx] = y_distrib[x_distrib_max_idx] - image_shape[0]

            if (input_image.shape[0] - y_distrib[y_distrib_max_idx]) < image_shape[0]:
                #print(input_image.shape[0] - y_distrib[y_distrib_max_idx])
                #print('y_max')
                #print(y_distrib_max_idx)
                #print(y_distrib[y_distrib_max_idx])
                #print(input_image.shape[0])
                y_distrib[y_distrib_max_idx] = y_distrib[y_distrib_max_idx] - image_shape[0]
                #print(y_distrib[y_distrib_max_idx])
                #if (input_image.shape[1] - x_distrib[y_distrib_max_idx]) < image_shape[1]:
                #    x_distrib[y_distrib_max_idx] = x_distrib[y_distrib_max_idx] - image_shape[1]
            # loop for number of batches
            for batch in range(num_batches):
                for patch_center in zip(x_distrib, y_distrib):
                    gt_images = []
                    count += 1
                    patch_name, _ = image_name.split('.')
                    patch_name = patch_name  + '_' + str(count)
                    # get the current patch center
                    x_coordi, y_coordi = patch_center
                    #print(patch_center)
                    # crop the patch from the center
                    patch_image_target = gt_image[y_coordi-image_shape_y:y_coordi+image_shape_y,
                        x_coordi-image_shape_x:x_coordi+image_shape_x]
                    # resize the image to have the given image_shape
                    patch_image_target = cv2.resize(patch_image_target, dsize=(image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_CUBIC)
                    # white background and black segments (background class)
                    _, patch_image_target_inv = cv2.threshold(patch_image_target, 127, 255,
                        cv2.THRESH_BINARY)
                    patch_image_target_inv = patch_image_target_inv
                    gt_images.append(patch_image_target_inv)
                    _, patch_image_target = cv2.threshold(patch_image_target, 127, 255,
                        cv2.THRESH_BINARY_INV)
                    patch_image_target = patch_image_target
                    gt_images.append(patch_image_target)
                    patch_image_input = input_image[y_coordi-image_shape_y:y_coordi+image_shape_y,
                        x_coordi-image_shape_x:x_coordi+image_shape_x]
                    patch_image_input = cv2.resize(patch_image_input, dsize=(image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_CUBIC)
                    #images.append(patch_image_input)
                    #gt_images.append(patch_image_target)
                    images = np.expand_dims(patch_image_input, axis=0)
                    gt_images = np.array(gt_images)
                    gt_images = np.expand_dims(gt_images, axis=0)
                    images = np.transpose(images, (0, 3, 1, 2))
                    #gt_images = np.expand_dims(gt_images, axis=1)
                    images_tensor = torch.from_numpy(images)
                    gt_images_tensor = torch.from_numpy(gt_images)
                    #print(patch_image.shape)
                    #cv2.imwrite(os.path.join(image_folder, 'image_test_patch', patch_name + '_target' + '.png'),
                    #            patch_image_target)
                    #cv2.imwrite(os.path.join(image_folder, 'image_test_patch', patch_name + 'input_' +  '.png'),
                    #            patch_image_input)
                    yield images_tensor.to(device, torch.float), \
                            gt_images_tensor.to(device, torch.float)
            #cv2.imwrite(os.path.join(image_folder, 'image_test', image_name), input_image)
    return batch_generator


def parser_xml(file_path):
    # get the list of xml files
    #file_list = glob.glob(os.path.join(directory, '*.xml'))
    coord_dic = dict()
    #for i in range(2):
    file_name = file_path
    xml_tree = ET.parse(file_name)
    root = xml_tree.getroot()
    count = 0 # counter for coordinates sets
    for f in root.iter():
        if f.tag == '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page':
            file_name = f.attrib['imageFilename']
    for t in root.iter(
        '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine'):
        #print(t.tag)
        count += 1
        current_coord = 'coordi_' + str(count)
        for c in t.iter(
                '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords'):
            coord_list = []
            point_pairs = c.attrib['points'] # get the all the coordi of the current textline
            point_str_list = point_pairs.split(' ')
            # make an int list of the coordi of the current textline
            for ele in range(len(point_str_list)):
                coord_pair_str = point_str_list[ele].split(',')
                coord_pair_int = list(map(int, coord_pair_str))
                #print(coord_pair_int)
                coord_list.append(coord_pair_int)
            #print('')
        # update the coord_dic with current textline and the corresponding coordinates
        coord_dic.update({current_coord: coord_list})
    #print(coord_dic)
    return file_name, coord_dic

def save_prediction(predicted_image, input_image, count):
    # for the input image
    input_image = input_image.cpu()
    input_image = input_image.detach().numpy()
    input_image = input_image.transpose((0, 2, 3, 1))
    input_image = input_image.squeeze(0)
    input_image = np.uint8(input_image * 255.0)
    cv2.imwrite('./data/ICDAR2017_simple/input_target_data/test_outputs/' + str(count) + '_input' + '.png',
    input_image)

    predicted_image = predicted_image.cpu()
    predicted_image = predicted_image.detach().numpy()
    predicted_image = predicted_image.transpose((0, 2, 3, 1))
    predicted_image = predicted_image.squeeze(0)
    predicted_image = predicted_image[:, :, 1]
    predicted_image = np.where(predicted_image > 0.5, 1, 0)
    predicted_image = np.uint8(predicted_image * 255.0)
    cv2.imwrite('./data/ICDAR2017_simple/input_target_data/test_outputs/' + str(count) + '_predicted' + '.png',
    predicted_image)
#lis, dic = parser_xml('./data/ICDAR2017_simple/input_target_data/xml_info/00000182.xml')
#print(lis)
#print(dic)
'''
gen = data_generator('./data/ICDAR2017_simple/input_target_data', image_shape=[300,300],
                     device=None)
for b in gen(20, 2):
    print(b)
'''
