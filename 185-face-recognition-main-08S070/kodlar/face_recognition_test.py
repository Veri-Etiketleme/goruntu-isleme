import face_recognition
import numpy as np
import cv2

# 參考：http://discoverbigdata.blogspot.com/2018/02/facerecognitionapi.html
'''這支程式主要在試用 face_recognition 套件'''

def draw(fileName, face_locations, face_landmarks):
    # 九組臉部特徵
    facial_features = [ 'chin',             # 下巴      17  點
                        'left_eyebrow',     # 左眉毛    5   點
                        'right_eyebrow',    # 右眉毛    5   點
                        'nose_bridge',      # 鼻樑      4   點
                        'nose_tip',         # 鼻尖      5   點
                        'left_eye',         # 左眼      6   點
                        'right_eye',        # 右眼      6   點
                        'top_lip',          # 上嘴唇    12  點
                        'bottom_lip']       # 下嘴唇    12  點
    
    img = cv2.imread(fileName)
    startY, endX, endY, startX = face_locations[0]
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 3)
    
    for landmarks in face_landmarks:
        # 印出九組特徵座標點
        for facial_feature in facial_features:
            print("臉部特徵({})座標點：{}".format(facial_feature, landmarks[facial_feature]))
        
        # 將九組特徵畫在原圖
        for facial_feature in facial_features:
            cv2.polylines(img, [np.array(landmarks[facial_feature])], False, (255,255,255), 2)

    name = fileName.split("/")[-1].split(".")[0]
    cv2.putText(img, name, (startX , endY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2) 
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    fileName = "face_dataset/self/cyt.png"
    picture_of_me = face_recognition.load_image_file(fileName)
    my_face_locations = face_recognition.face_locations(picture_of_me, number_of_times_to_upsample=1, model='hog')
    my_face_encodings = face_recognition.face_encodings(picture_of_me, known_face_locations=my_face_locations, num_jitters=10)[0] # 因為知道這張照片只有一張臉所以index=0
    my_face_landmarks = face_recognition.face_landmarks(picture_of_me, face_locations=my_face_locations)
    # draw(fileName, my_face_locations, my_face_landmarks)
    # print("照片共找出 {} 張臉。".format(len(my_face_locations)))
    # print("(上、右、下、左) = {}".format(my_face_locations[0]))
    
    fileName = "face_dataset/self/cyt2.jpg"
    unknown_picture = face_recognition.load_image_file(fileName)
    unknown_face_locations = face_recognition.face_locations(unknown_picture, number_of_times_to_upsample=1, model='hog')
    unknown_face_encodings = face_recognition.face_encodings(unknown_picture, known_face_locations=unknown_face_locations, num_jitters=10)[0]
    unknown_face_landmarks = face_recognition.face_landmarks(unknown_picture, face_locations=unknown_face_locations)
    # draw(fileName, unknown_face_locations, unknown_face_landmarks)
    # print("照片共找出 {} 張臉。".format(len(unknown_face_locations)))
    # print("(上、右、下、左) = {}".format(unknown_face_locations[0]))

    # 進行比對並顯示結果
    results = face_recognition.compare_faces([my_face_encodings], unknown_face_encodings, tolerance=0.4)
    euclidean_distance = face_recognition.face_distance([my_face_encodings], unknown_face_encodings)
    
    if results[0] == True:
        print("It's me! Euclidean distance is: {}".format(euclidean_distance))
    else:
        print("It's not me! Euclidean distance is: {}".format(euclidean_distance))
    