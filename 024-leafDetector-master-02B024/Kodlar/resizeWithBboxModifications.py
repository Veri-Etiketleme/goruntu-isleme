'''
Author : Kunchala Anil  
Email : anilkunchalaece@gmail.com
Date : 8 Feb 2019

This script will resize all the images with along with their respective bounding boxes

'''

import os,shutil
import cv2
from xml.etree import ElementTree

class ResizeWithBbox:

    def __init__(self,srcImgDir,desImgDir,desImgSize):
        
        self.srcImgDir = srcImgDir
        self.desImgDir = desImgDir
        self.desImgHeight = desImgSize[0]
        self.desImgWidth = desImgSize[1]
        if os.path.isdir(self.desImgDir) :
            shutil.rmtree(self.desImgDir)
        os.mkdir(self.desImgDir)

    def resizeAllImagesInDir(self):
        totalFiles = os.listdir(self.srcImgDir)
        for fileName in totalFiles :
            print(fileName)
            # ignore .xml files
            if fileName.find('.xml') == -1 :
                self.resizeImgwithBbox(fileName)
    
    def resizeImgwithBbox(self,imgName):
        
        orgImgPath = os.path.join(self.srcImgDir+imgName)
        print("running for img : {}".format(orgImgPath))
        orgImgObj = cv2.imread(orgImgPath)
        orgImgHeight = int(orgImgObj.shape[0])
        orgImgWidth = int(orgImgObj.shape[1])

        self.imgResizeHeightRatio = self.desImgHeight / orgImgHeight
        self.imgResizeWidthRatio = self.desImgWidth /orgImgWidth

        print("resizing the image : {} with height {} and width {} to resize height ratio {} \
         and width ratio {}".format(orgImgPath,orgImgHeight,orgImgWidth,self.imgResizeHeightRatio,self.imgResizeWidthRatio))
        resizedImgObj = cv2.resize(orgImgObj,(self.desImgHeight,self.desImgWidth))
        cv2.imwrite(os.path.join(self.desImgDir,imgName),resizedImgObj)

        self.reWriteXmlFile(imgName)

    def reWriteXmlFile(self,imgName) :
        srcFileName = os.path.join(self.srcImgDir,imgName).replace(".jpg",".xml")
        desFilePath = os.path.join(self.desImgDir,imgName).replace(".jpg",".xml")
        #opening file to write and writing the commom values
        fHandler = open(desFilePath,'w')
        fHandler.writelines(""" <annotation> 
                                <folder>eveningData-lenovo</folder> 
                                <filename>{}</filename> 
                                <path>{}</path> 
                                <source> 
                                    <database>Unknown</database> 
                                </source>
                                <size>
                                    <width>{}</width>
                                    <height>{}</height>
                                    <depth>3</depth>
                                </size>""".format(imgName,srcFileName.replace('.xml','.jpg'),self.desImgWidth,self.desImgHeight))
        xmlRoot = ElementTree.parse(srcFileName).getroot()
        for element in xmlRoot.findall('object'):
            #ref - https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
            bbox = element.find('bndbox')
            xmin = str(int(int(bbox.find('xmin').text) * self.imgResizeWidthRatio))
            ymin = str(int(int(bbox.find('ymin').text) * self.imgResizeHeightRatio))
            xmax = str(int(int(bbox.find('xmax').text) * self.imgResizeWidthRatio))
            ymax = str(int(int(bbox.find('ymax').text) * self.imgResizeHeightRatio))
            truncated = element.find('truncated').text
            labelName = element.find('name').text

            strToReplace = """
                            <object>
                                <name>{}</name>
                                <pose>Unspecified</pose>
                                <truncated>{}</truncated>
                                <difficult>0</difficult>
                                <bndbox>
                                    <xmin>{}</xmin>
                                    <ymin>{}</ymin>
                                    <xmax>{}</xmax>
                                    <ymax>{}</ymax>
                                </bndbox>
                            </object> """.format(labelName,truncated,xmin,ymin,xmax,ymax)
            fHandler.writelines(strToReplace)
        # close annoitations and fileHandler
        fHandler.write('</annotation>')
        fHandler.close()


    

def main():
    SRC_IMG_DIR = "/home/ic/Documents/leafData/org/"
    DES_IMG_DIR = "/home/ic/Documents/leafData/resize640/"
    # used for both tensorflow and digits 
    # based on article Exploring SpaceNet dataset using Digits
    DES_IMG_HEIGHT = 640
    DES_IMG_WIDTH = 640
    RESIZE_RATIO = (DES_IMG_HEIGHT,DES_IMG_WIDTH)
    imgResize = ResizeWithBbox(SRC_IMG_DIR,DES_IMG_DIR,RESIZE_RATIO)
    imgResize.resizeAllImagesInDir()

if __name__ == '__main__' :
    main()