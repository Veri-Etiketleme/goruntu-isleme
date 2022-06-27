'''
Author : Kunchala Anil
Email : anilkunchalaece@gmail.com
Date : Feb 2 2019

This is used to convert xml files created using labelImg to Kitti format which 
are supported by NVIDIA digits platform
'''

import os,shutil
from xml.etree import ElementTree

class LabelImg2Kitti:

    def __init__(self,rootDir) :
        self.rootDir = rootDir
        self.srcDir = self.rootDir + '/images'
        self.desDir = self.rootDir + '/labels'
        #remove the destination directory if it exists
        if os.path.isdir(self.desDir) :
            shutil.rmtree(self.desDir)
        os.mkdir(self.desDir)
        self.xmlFileList = self.getAllXMLFiles()
        self.xmlData = list()

    def convertAllFilesInDir(self) :
        for fileName in self.xmlFileList :
            self.convertXMLFileToKitti(self.srcDir+'/'+fileName)
            self.writeToFile(self.desDir+'/'+fileName.replace('.xml','.txt'))

    def getAllXMLFiles(self) :
        '''
        This function is used to get all the xml files in the specified directory
        '''
        allFiles = os.listdir(self.srcDir)
        xmlList = []
        for file in allFiles :
            if file.find('.xml') != -1 : 
                xmlList.append(file)

        print('total no of files in src directory : ' + str(len(xmlList)))
        return xmlList
    
    def convertXMLFileToKitti(self,fileName):
        #reintialize the list
        self.xmlData = []
        xmlRoot = ElementTree.parse(fileName).getroot()
        for element in xmlRoot.findall('object'):
            bbox = element.find('bndbox')
            xmin = str(float(bbox.find('xmin').text))
            xmax = str(float(bbox.find('xmax').text))
            ymin = str(float(bbox.find('ymin').text))
            ymax = str(float(bbox.find('ymax').text))
            truncated = str(float(element.find('truncated').text))
            labelName = element.find('name').text
            self.xmlData.append({
                'xmin' : xmin,
                'ymin' : ymin,
                'ymax' : ymax,
                'xmax' : xmax,
                'truncated' : truncated,
                'labelName' : labelName
            })
            
    
    
    def writeToFile(self,fileName):
        '''
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better
        '''
        self.defineIgnoredElemets()
        fHandler = open(fileName,'w')
        for ele in self.xmlData : 
            fHandler.write(ele['labelName'])
            fHandler.write(' ')
            fHandler.write(ele['truncated'])
            fHandler.write(' ')
            fHandler.write(self.occluded)
            fHandler.write(' ')
            fHandler.write(self.alpha)
            fHandler.write(' ')
            fHandler.write(ele['xmin'])
            fHandler.write(' ')
            fHandler.write(ele['ymin'])
            fHandler.write(' ')
            fHandler.write(ele['xmax'])
            fHandler.write(' ')
            fHandler.write(ele['ymax'])
            fHandler.write(' ')
            fHandler.write(self.dimHeight)
            fHandler.write(' ')
            fHandler.write(self.dimWidth)
            fHandler.write(' ')
            fHandler.write(self.dimLength)
            fHandler.write(' ')
            fHandler.write(self.locationX)
            fHandler.write(' ')
            fHandler.write(self.locationY)
            fHandler.write(' ')
            fHandler.write(self.locationZ)
            fHandler.write(' ')
            fHandler.write(self.rotationY)
            fHandler.write("\n")
        fHandler.close()

    def defineIgnoredElemets(self):
        '''
        Following parameters require for kitti but ignored by detectnet
        '''
        self.occluded = str(int(0))
        self.alpha = str(float(0)) 
        self.dimHeight = str(float(0))
        self.dimWidth = str(float(0)) 
        self.dimLength = str(float(0)) 
        self.locationX = str(float(0))
        self.locationY = str(float(0)) 
        self.locationZ = str(float(0))
        self.rotationY = str(float(0))


def main() :
    SRC_DIR = 'multiLeafResized'
    convertToKitti = LabelImg2Kitti(SRC_DIR) 
    # convertToKitti.convertXMLFileToKitti('singleLeaf/IMG_20190201_165039.xml') 
    # convertToKitti.writeToFile('singleLeaf/IMG_20190201_165039.txt')
    convertToKitti.convertAllFilesInDir()

if __name__ == "__main__" :
    main()