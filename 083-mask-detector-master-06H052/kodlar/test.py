from fastai.vision import *
import cv2 as cv
import numpy as np


face_detector_prototxt_path = 'models/deploy.prototxt'
face_detector_weights_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
mask_detector_path = 'models'

if __name__ == '__main__':
    learn = load_learner(mask_detector_path, 'resnet50_40')
    face_net = cv.dnn.readNet(face_detector_prototxt_path,
                             face_detector_weights_path)

    # Detection on image
    # img = open_image('images/1.jpg')
    # print(learn.predict(img))

    # Detection on video
    vs = cv.VideoCapture("videos/2.mp4")

    while True:
        _, frame = vs.read()
        frame = cv.resize(frame, (400, 225), interpolation=cv.INTER_LINEAR)
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300),
                                    (104.0, 177.0, 123.0))
        face_net.setInput(blob)

        detections = face_net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            (h, w) = frame.shape[:2]

            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX-20), max(0, startY-20))
                (endX, endY) = (min(w - 1, endX+20), min(h - 1, endY+20))

                face = frame[startY:endY, startX:endX]
                face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                face = cv.resize(face, (224, 224))
                img = Image(pil2tensor(face, dtype=np.float32).div_(255))

                label, _, confidence = learn.predict(img)

                if confidence[0] > 0.6 or confidence[1] > 0.6:
                    color = (0, 255, 0) if str(
                        label) == "mask" else (0, 0, 255)

                    cv.putText(frame, str(label), (startX, endY + 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv.rectangle(frame, (startX, startY),
                                 (endX, endY), color, 2)

                cv.imshow("Frame", frame)
                cv.waitKey(1)
