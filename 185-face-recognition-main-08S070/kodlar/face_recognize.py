import os
import time
import cv2
import time
import numpy as np
import utils.utils as utils
from net.inception import InceptionResNetV1
'''
# 臉部對齊
import dlib
from imutils.face_utils import FaceAligner

# 年紀國籍性別偵測 ./facial/face_video_detect.py
from facial.face_video_detect import facial_detcet

# 口罩偵測 ./FaceMaskDetection/keras_infer.py
from FaceMaskDetection.keras_infer import inference
'''
class face_rec():
    def __init__(self):
        '''
        Step1. Create fanet and facenet_model model
        Step2. Get face encode
        '''
        
        # fanet -           face detect
        # facenet_model -   turn face into 128-d vector
        startTime = time.time()
        print('Create fanet...')
        self.fanet = cv2.dnn.readNetFromTensorflow("./pretrained_model/opencv_face_detector_uint8.pb", "./pretrained_model/opencv_face_detector.pbtxt")       
        
        print('Create facenet_model...')
        self.facenet_model = InceptionResNetV1()
        self.facenet_model.load_weights('./pretrained_model/facenet_keras.h5')
        endTime = time.time()
        print("載入模型時間: {}s".format(endTime-startTime))
        
        
        # Get face encode from images in face_dataset
        startTime = time.time()
        self.known_face_encodings = []
        self.known_face_names = []
        self.get_face_encode()
        endTime = time.time()
        print("計算face_encoding時間: {}s".format(endTime-startTime)) # 896.7800765037537s # 1.4956989288330078s

    def get_face_encode(self):
        '''
        Step1. Use "fanet" to detect face and crop it. 
        Step2. Put the cropped image to "facenet_model" and transfer to 128-d vector
        '''

        face_list = os.listdir("face_dataset/self")
        for face in face_list:
            name = face.split(".")[0]
            img = cv2.imread("./face_dataset/self/" + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect face
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.fanet.setInput(blob)
            detections = self.fanet.forward()
            
            h, w, _ = img.shape
            (startX, startY, endX, endY) = -100, -100, -100, -100
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # 當檢測到人臉置信度大於80%
                if confidence > 0.8:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
            # 框出的位置
            rectangles = (startX, startY, endX, endY)

            # Transfer to 128-d vector
            crop_img = img[int(rectangles[1]):int(rectangles[3]), int(rectangles[0]):int(rectangles[2])]
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)        
        
    def detect_img(self, image):
        '''Detect face and send it to recognition'''
       
        net = self.fanet
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        h, w, _ = image.shape
        (startX, startY, endX, endY) = -100, -100, -100, -100
        start_time= time.time()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # 當偵測到人臉 信心度大於80%
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")   

                '''
                # 檢測人臉特徵(年紀國籍性別)
                face_locations=[]
                if startX != -100:
                    face_locations=[(startY, endX, endY,startX)]
                image=facial_detcet(image,face_locations)

                # 檢測口罩
                image=inference(image,startX,endY,0.8)
                ''' 
                
                image = self.recognize(image, startX, startY, endX, endY)
            
        end_time = time.time()
        print("辨識時間: {}".format(end_time - start_time))
        return  image
    
    def recognize(self, image, startX, startY, endX, endY):
        '''
        Get the face encode first
        Find which has the shortest distance from it
        '''

        draw_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rectangles = (startX, startY, endX, endY)

        # 擷取圖像 & 計算face encode
        crop_img = draw_rgb[int(rectangles[1]):int(rectangles[3]), int(rectangles[0]):int(rectangles[2])]
        crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
        face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)
        
        # 將剛計算出來的 encode 與 face_dataset 中的人臉進行比對
        # 並計算分數，與 face_dataset 中每一張臉的距離分數
        # 取出與這張臉最接近的名稱
        matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.6)
        name = "Unknown"
        face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
        
        # 畫框 + 寫名字
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 3)
        cv2.putText(image, name, (startX , endY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2) 
        
        return image

if __name__ == "__main__":
    '''測試功能完整性'''
    face_net = face_rec()
    video_capture = cv2.VideoCapture(0)
    
    if video_capture.isOpened() == False:
        print("啟動相機")
        video_capture.open()
    
    # 設定影像的尺寸大小，更改攝影機畫面解析度
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    while True:
        status, frame = video_capture.read()

        face_net.recognize(frame, 490, 330, 790, 630) 
        cv2.imshow('Video', frame)

        #  按下 q 離開迴圈
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
