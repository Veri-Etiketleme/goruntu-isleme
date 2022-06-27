from Webcam import Webcam
from Human_detector import Human_Detector
from Mask_detector import Mask_Detector, MaskConvNet
from Face_detector import Face_detector
from PIL import Image as I
import time
import cv2

webcam = Webcam()
human_detector = Human_Detector()
face_detector = Face_detector()
mask_Detector = Mask_Detector("Mask_CNN.pth")


cv2.namedWindow("viewport")


#main loop

while(1):
    frame, pil_in = webcam.get_frame()
    safe = True
    human = False
    
    list_of_coords, list_of_humans = human_detector.detect(pil_in)

    if(len(list_of_humans) > 0):
        human = True

    for (x_h,y_h,w_h,h_h) in list_of_coords:

        (x_h,y_h,w_h,h_h) = (int(x_h) ,int(y_h) , int(w_h), int(h_h))
        safe = False

        frame = cv2.rectangle(frame,(x_h,y_h),( x_h + w_h ,y_h + h_h),(255,0,0),2)

       
        faces = face_detector.detect( frame[y_h:(y_h + h_h), x_h:(x_h+w_h)] )
        

        for (x,y,w,h) in faces:

            i = pil_in.crop((x + x_h, y + y_h, x+w + x_h , y+h + y_h))

            
            prediction = mask_Detector.classify(i)
            

            if(prediction == 0):
                frame = cv2.rectangle(frame,(x+x_h,y+y_h),( x+x_h + w ,y+y_h + h),(0,255,0),2)
                safe = True
            else:
                frame = cv2.rectangle(frame,(x+x_h,y+y_h),( x+x_h + w ,y+y_h + h),(0,0,255),2)
                

    if(human == False):
        frame = cv2.putText(frame,  "No human detected. Door closed", (10,25), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0), 2, cv2.LINE_AA )
    else:
        if(safe == True):

            frame = cv2.putText(frame,  "Human detected. All faces wearing mask. Door Opened", (10,25), cv2.FONT_HERSHEY_PLAIN ,1, (255,0,0), 2, cv2.LINE_AA )
        else:

            frame = cv2.putText(frame,  "Human detected. Face with no mask detected. Door closed", (10,25), cv2.FONT_HERSHEY_PLAIN ,1, (255,0,0), 2, cv2.LINE_AA )

    
    cv2.imshow("viewport", frame)

    cv2.waitKey(10)



cv2.destroyWindow("preview")
