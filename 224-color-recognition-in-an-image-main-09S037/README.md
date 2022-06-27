# color-recognition-in-an-image
Problem Statement 
 
To identify the colours in the images of sample . 
To create a k-means clustering model that will make colour data sample which can be further used to capture the colour of testing  images  . 
To create clusters of all the RGB colours and then using them for further images . 

we used to get millions of trillions of images these days and many times we need to go specific with the search 
requiring some specific colour and types of images for this same things we are making a machine learning model where our 
learner will learn the colours with the help all the algorithms and packages used and then conclude by itself which colour is in 
the image and it will differentiate the images on the basis of maximum percentage of a particular colour present in the images. 
we have used some images as sample out of which we are first  using some  for learning  and then some for  testing our model.

this project is being run in google colab here we are importing images sample from google drive for this we have to mount our drive with google colab
then import all the necessary libraries.
flow of program is->
import an image from samples
         ||
         \/
convert image into array using numpy
         ||
         \/
extract 10 major colors from the image
        ||
        \/
get hex values of all the colors
        ||
        \/
now compare the color hex value of color entered by the user and the hex value of major colors found in images
        ||
        \/
return the images which have colors match 
