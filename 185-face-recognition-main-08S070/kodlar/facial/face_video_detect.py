
'''
demo for camera
'''

import numpy as np
import cv2
import face_recognition
from facial.face import Face
from facial.utils import putText
from facial.utils import preprocess_input

model = Face(train=False)
model.load_weights('./facial/face_weights/face_weights.16-val_loss-4.42-val_age_loss-3.76-val_gender_loss-0.48-val_race_loss-0.18.utk.h5')

gender_labels = ['Male', 'Female']
race_labels = ['Foreigner', 'Asian']
#https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
age_labels = np.reshape(np.arange(1, 94), (93,1))


def facial_detcet(frame,face_locations):

#ret, frame = cap.read()

    #face_locations = face_recognition.face_locations(frame, model='cnn')
    
    if len(face_locations) > 0:
        face_batch = np.empty((len(face_locations), 200, 200, 3))

        # add face images into batch
        for i,rect in enumerate(face_locations):
            face_img = frame[rect[0]:rect[2], rect[3]:rect[1], :]
            face_img = cv2.resize(face_img, (200, 200))
            face_batch[i, :, :, :] = face_img
        
        face_batch = preprocess_input(face_batch)
        preds = model.predict(face_batch)

        preds_ages = preds[0]
        preds_genders = preds[1]
        preds_races = preds[2]
        
        # dispaly on srceen
        for rect, age, gender, race in zip(face_locations, preds_ages, preds_genders, preds_races):
            #cv2.rectangle(frame, (rect[3], rect[0]), (rect[1], rect[2]), (255, 0, 0), 2)
            age = np.expand_dims(age, 0)
            #print(age)
            # https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
            age_data = int(age.dot(age_labels).flatten())
            gender_index = np.argmax(gender)
            race_index = np.argmax(race)
            frame = putText(frame, 'gender: {0}'.format(gender_labels[gender_index]), (255, 0, 0), (rect[3], rect[0]-32), size=17)
            frame = putText(frame, 'race: {0}'.format(race_labels[race_index]), (255, 0, 0), (rect[3], rect[0]-50), size=17)
            frame = putText(frame, 'age: {0}'.format(age_data+10), (255, 0, 0), (rect[3], rect[0]-68), size=17)
    
    return frame
    

