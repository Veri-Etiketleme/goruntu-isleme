# Image Segmentation for AB4
Craioveanu Sergiu-Ionut

21-April-2020

# Implementation

Task: To implement a method of image segmentation that helps optimize contours of objects. To be more specific, identify certain parts of brain images from CT-scans, and improve upon the manual selection made by doctors.

The structure of the code is divided as follows:
* Imports, working path
* Data Manipulation
* Dataset
* Model

I will delve into further detail regarding each chapter of the code below.
## Important mention: The solution is not functional due to data restraints, but represents an approach which has the potential of being highly competitive. Such approaches are seen in Kaggle Competitions among the top 10%, if not top 1%. The efficiency of the code should not be judged by the result, but by the potential it presents itself with.


## Intro

We import the modules needed, such as *fastai* and *os*. We set the working path as the current directory. In this respect, it is platform independent, therefore scalable.

## Data Manipulation

We split the data in labels and images, as per usual with typical Image Segmentation problem approaches. 

The *images* folder contains pure CT-Scans, whereas the *labels* folder contains the mask of corresponding images.

The *classes.txt* document contains a list of all the existing classes within the CT-scans. They are found in the documentation and are based upon the Hounsfield values of DICOM images.

For ease of work, we define a function that takes the path of our image and associates the corresponding label. This way, we can just find the corresponding label of an image by sending the pathname as a parameter to the function.

We also check that images and labels are read correctly by displaying them.

## Dataset

Here we make some final decisions regarding the dataset, before we send it to our model.

We split the data into training/validation, based upon the *valid.txt*. This makes it so that the split is predefined and not randomized. It offers consistency. As mentioned earlier, we use a function to obtain the labels from the pathname of source images. This generates a source object.

We create our data using our source object, and we normalize it using imagenet_stats because of the model chosen: Resnet34 - which has been pretrained with imagenet images. We also choose a batch size of 2, but these hyper-parameters can be modified accordingly.

We perform additional verifications to make sure dimensions match.

## Model

Although the code is quite self-explanatory, I will add a few more details, that explain why the solution fails, in essence. For the model to properly learn, it requires *AT THE VERY LEAST* 100 samples per category/class. Seeing as this is a high precision task, the data necessary is perhaps orders of magnitude higher. As a reference, we have at least 7 different classes and multiple segments of brain parts. 

?? I am unsure of the types of data augmentations that might have been performed to multiply our small dataset. The nature of my doubt comes from the special properties that DICOM images present themselves with. ??

The idea is to plot our learning rate through the evolution of epochs. This lets us optimize out parameters for weight decay and learning rate, to name a few. Unfortunately, with the data I had, such a thing was impossible. 

?? An alternative might have been to try a different or smaller neural network. In my opinion, that defeats the purpose of attempting to create a high precision solution. So, the eternal problem of High complexity - More data arises. ??

After we supposedly train our model over the epochs and find 'optimal' hyper-parameters, we can save our model. This allows for further improvement, yet again, offering scalability.

## ROOM FOR IMPROVEMENT

* MORE DATA, obviously. More CT-scans and their mask provided by the doctors.

* After the second stage has been completed, we enter the fine tuning domain. We can try and unfreeze the model, and retrain it. Alternatively, we can try and unfreeze only the last few layers and retrain those, with different image sizes. As far as the model is concerned, different sized images are different images.
* Further modification of learning rates, using a more sophisticated neural net such as RESNET50.
* Add .cuda() for the highly computational operations, switching the processing power upon the GPU, instead of the CPU. Needless to say, it allows for faster experimentation. 

All the above steps should really bring highly competitive results.

