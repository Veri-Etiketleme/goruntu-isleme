'''
Author : Kunchala Anil
Email : anilkunchalaece@gmail.com
Date : Feb 2 2019

This script is used to devide the data set in to training and validation
dataset is in kitti format

Create two dirs in Root Folder
    images and labels and save those files in respective directories and run script

'''
import os,random,shutil

class DevideDataSet:
    def __init__(self,rootDir) :
        self.rootDir = rootDir
        self.trainDir = self.rootDir + '/train'
        self.evalDir = self.rootDir + '/validate'
        self.trainImagesDir = self.trainDir + '/images'
        self.trainLabelsDir = self.trainDir + '/labels'
        self.evalImagesDir = self.evalDir + '/images'
        self.evalLabelsDir = self.evalDir + '/labels'
        listOfDirectoriesToCreate = [
            self.trainDir,self.evalDir,self.trainImagesDir,self.trainLabelsDir,self.evalImagesDir,self.evalLabelsDir
        ]
        self.createDirectoryStrecture(listOfDirectoriesToCreate)
    
    def createDirectoryStrecture(self,dirList):
        for directory in dirList :
            # if directory exists remove it
            if os.path.isdir(directory) :
                shutil.rmtree(directory)
            os.mkdir(directory)
    

    def devideData(self,evalPercentage):
        totalFiles = os.listdir(self.rootDir+'/images')
        totalNoOfImages = len(totalFiles)
        evalImagesToTake = int((totalNoOfImages*evalPercentage))
        
        #get the random set based on evalImages
        randomEvalSet = random.sample(range(0,totalNoOfImages),evalImagesToTake)

        for fileIndex in range(len(totalFiles)) :
            if fileIndex in randomEvalSet :
                shutil.copy2(self.rootDir+'/images/'+totalFiles[fileIndex], self.evalImagesDir)
                shutil.copy2(self.rootDir+'/labels/'+totalFiles[fileIndex].replace('.jpg','.txt'),self.evalLabelsDir)
            else :
                shutil.copy2(self.rootDir+'/images/'+totalFiles[fileIndex], self.trainImagesDir)
                shutil.copy2(self.rootDir+'/labels/'+totalFiles[fileIndex].replace('.jpg','.txt'),self.trainLabelsDir)


def main():
    ROOT_DIR = 'multiLeafResized'
    EVAL_PERCENTAGE = 0.30
    data = DevideDataSet(ROOT_DIR)
    data.devideData(EVAL_PERCENTAGE)

if __name__ == '__main__' :
    main()