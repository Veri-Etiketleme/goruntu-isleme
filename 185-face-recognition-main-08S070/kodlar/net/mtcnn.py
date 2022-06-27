import cv2
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from utils import utils
# from keras.utils import plot_model


# 粗略取得人臉框，是否有人臉
def create_Pnet(weight_path):
    inputs = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性。
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# 取得精準人臉框
def create_Rnet(weight_path):
    inputs = Input(shape=[24, 24, 3])
    # 24,24,3 -> 22,22,28 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    # 11,11,28 -> 9,9,48 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)

    # 128 -> 2
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 128 -> 4
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# 獲得人臉特徵點
def create_Onet(weight_path):
    inputs = Input(shape = [48,48,3])
    # 48,48,3 -> 46,46,32 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 21,21,64 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)

    # 3,3,128 -> 128,12,12
    x = Permute((3,2,1))(x)
    x = Flatten()(x)

    # 1152 -> 256
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2 
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    # 256 -> 4 
    bbox_regress = Dense(4,name='conv6-2')(x)
    # 256 -> 10 
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('pretrained_model/pnet.h5')
        self.Rnet = create_Rnet('pretrained_model/rnet.h5')
        self.Onet = create_Onet('pretrained_model/onet.h5')
        # plot_model(self.Pnet, to_file='Pnet.png', show_shapes=True)
        # plot_model(self.Rnet, to_file='Rnet.png', show_shapes=True)
        # plot_model(self.Onet, to_file='Onet.png', show_shapes=True)
    
    def detectFace(self, img, threshold):
        # 色彩歸一化: -1~1
        # img格式: (height, width, 3)
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape

        # 做成 image pyramid 計算影像縮放大小
        scales = utils.calculateScales(img)

        # 粗略計算人臉框 - Pnet部分
        out = []
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = np.expand_dims(scale_img, 0)
            ouput = self.Pnet.predict(inputs)
            ouput = [ouput[0][0], ouput[1][0]]
            out.append(ouput)
        
        
        # 取出每張圖的種類預測和回歸結果
        rectangles = []
        for i in range(len(scales)):
            cls_prob = out[i][0][:, :, 1]
            roi = out[i][1]

            # 取出縮放後的寬高
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)

            # 解碼
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        
        # 非極大抑制
        rectangles = np.array(utils.NMS(rectangles, 0.7))

        if len(rectangles) == 0:
            print('Cannot find any face in this frame.')
            return rectangles
        
        # ------------------------------------------------ #
        # 精確計算人臉框 - Rnet部分
        predict_24_batch = []
        for rectangle in rectangles:
            # 利用粗略座標(Pnet結果)，在原圖上擷取。
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            
            # resize 到 24*24
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        cls_prob, roi_prob = self.Rnet.predict(np.array(predict_24_batch))

        # 解碼
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            print('Cannot find any face in this frame.')
            return rectangles
        
        # ------------------------------------------------ #
        # 計算人臉特徵點 - Onet部分
        predict_batch = []
        for rectangle in rectangles:
            # 利用精確座標(Rnet結果)，從原圖擷取
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            
            # resize 成 48*48 
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        cls_prob, roi_prob, pts_prob = self.Onet.predict(np.array(predict_batch))
        
        # 解碼
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        return rectangles