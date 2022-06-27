import cv2
import numpy as np
import utils.utils as utils
from net.mtcnn import mtcnn

if __name__ == "__main__":
    '''人臉對齊(眼睛呈水平)，並未與 main.py 連接。可以先從這支程式看看如何做 face align'''
    # 門檻值
    threshold = [0.5,0.7,0.8]

    # 建立mtcnn模型
    mtcnn_model = mtcnn()

    # 讀取照片
    img = cv2.imread('face_dataset/self/timg.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 轉正方形
    rectangles = mtcnn_model.detectFace(img, threshold)
    rectangles = utils.rect2square(np.array(rectangles))

    if len(rectangles) == 0:
        print('Cannot find any face in this frame.')
    else:
        '''四張圖一起show，原圖、原圖+臉部特徵點、旋轉、旋轉+臉部特徵點'''
        for rectangle in rectangles:
            # 擷取影像(原圖)
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([rectangle[0], rectangle[1]])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            
            # (原圖+臉部特徵點)
            copy_img = crop_img.copy()
            for xy in landmark:
                x = int(xy[0])
                y = int(xy[1])
                copy_img = cv2.circle(copy_img, (x, y), radius=10, color=(255, 0, 0), thickness=-1)

            # 拼接 (原圖 & 原圖+臉部特徵點)
            Hori1 = np.concatenate((crop_img, copy_img), axis=1)
            


            # 利用人臉特徵點對齊(旋轉)
            crop_img, new_landmark = utils.Alignment_1(crop_img, landmark)

            # (旋轉 & 旋轉+臉部特徵點)
            copy_img = crop_img.copy()
            for xy in new_landmark:
                x = int(xy[0])
                y = int(xy[1])
                copy_img = cv2.circle(copy_img, (x, y), radius=10, color=(255, 0, 0), thickness=-1)

            # 拼接 (旋轉 & 旋轉+臉部特徵點)
            Hori2 = np.concatenate((crop_img, copy_img), axis=1)

            # 拼接 (Hori1 & Hori2)
            Verti = np.concatenate((Hori1, Hori2), axis=0)
            cv2.imshow("Totally", cv2.cvtColor(Verti, cv2.COLOR_RGB2BGR))

            cv2.waitKey(0)
            cv2.destroyAllWindows()
