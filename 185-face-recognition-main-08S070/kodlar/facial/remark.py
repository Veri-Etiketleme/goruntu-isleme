import os
import numpy as np
import cv2
import face_recognition
from face import Face
from utils import putText
from utils import preprocess_input
train_data_path = './train_data/'

model = Face(train=False)
model.load_weights('./face_weights/face_weights.07-val_loss-4.49-val_age_loss-3.62-val_gender_loss-0.50-val_race_loss-0.36.utk.h5')

gender_labels = ['Male', 'Female']
race_labels = ['Foreigner', 'Asian']
#https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
age_labels = np.reshape(np.arange(1, 94), (93,1))

fileList=os.listdir(train_data_path)
a=0
b=0
c=0
d=0
e=0
f=0
g=0
h=0
j=0
n=0
asi=0
fori=0
for i in fileList:
        
        f_items = str(fileList[n]).split('_')
        if int(f_items[0])>=1 and int(f_items[0])<=10:
            a+=1
        elif int(f_items[0])>=11 and int(f_items[0])<=20:
            b+=1
        elif int(f_items[0])>=21 and int(f_items[0])<=30:
            c+=1
        elif int(f_items[0])>=31 and int(f_items[0])<=40:
            d+=1
        elif int(f_items[0])>=41 and int(f_items[0])<=50:
            e+=1
        elif int(f_items[0])>=51 and int(f_items[0])<=60:
            f+=1
        elif int(f_items[0])>=61 and int(f_items[0])<=70:
            g+=1
        elif int(f_items[0])>=71 and int(f_items[0])<=80:
            h+=1
        elif int(f_items[0])>=81 :
            j+=1
        n+=1  
        
print (a,b,c,d,e,f,g,h,j)          
'''
#f = open('forwriteasian.txt','w')
for i in fileList:
        
        f_items = str(fileList[n]).split('_')
        oldname=train_data_path+ os.sep + fileList[n]
        
        if str(f_items[2])=="1":
            print(oldname)
            demo_image = cv2.imread(oldname)
            #cv2.imshow('image', demo_image)
            image_h, image_w = demo_image.shape[0], demo_image.shape[1]
            margin = 0.01
            face_locations = face_recognition.face_locations(demo_image, model='cnn')
      
            if len(face_locations) > 0:
                face_batch = np.empty((len(face_locations), 200, 200, 3))
            
               # add face images into batch
                for i,rect in enumerate(face_locations):
                   # crop with a margin
                    top, bottom, left, right = rect[0], rect[2], rect[3], rect[1]
                    top = max(int(top - image_h * margin), 0)
                    left = max(int(left - image_w * margin), 0)
                    bottom = min(int(bottom + image_h * margin), image_h - 1)
                    right = min(int(right + image_w * margin), image_w - 1)
            
                    face_img = demo_image[top:bottom, left:right, :]
                    face_img = cv2.resize(face_img, (200, 200))
                    face_batch[i, :, :, :] = face_img
                
                face_batch = preprocess_input(face_batch)
                preds = model.predict(face_batch)
            
                preds_ages = preds[0]
                preds_genders = preds[1]
                preds_races = preds[2]
                for rect, ageQQ, gender, race in zip(face_locations, preds_ages, preds_genders, preds_races):
                    gender_index = np.argmax(gender)
                    race_index = np.argmax(race)
           
            #print(str(j),str(gender_index),str(index))
            if str(race_index)== "0":
                print("Foreigner",oldname)
                f.write(oldname+"\n")
                
                
        #os.rename(oldname,newname) 
        #print(oldname,'======>',newname)
        n+=1

f.close()
'''
