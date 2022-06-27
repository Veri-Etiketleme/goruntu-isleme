from FrameIterator_test import test_frame_iterator
from FrameProcessing import frame_processor
import cv2
import numpy as np

# Extracting facial geometric features as a displacement between the reference frame and every consequent frame

def GFeatures(FRAMES_PATH, PREDICTOR_PATH):
    file_names = test_frame_iterator(FRAMES_PATH)
    fp = frame_processor(PREDICTOR_PATH)
    landmarks_num = [48, 45, 54, 57, 19, 24, 17, 21, 22, 26, 51, 36, 39, 37,
                     38, 41, 40, 42, 45, 43, 44, 47, 46]
    ref_coords = []
    GFeatures_collection = []

    for file_name in file_names:
        print('processing frame: ' + file_name)
        frame = cv2.imread(FRAMES_PATH + file_name)
        landmarks = fp.get_all_landmarks(frame)
        selected_landmarks = fp.selected_landmarks(landmarks_num, frame)
        current_coords = []

        for i in selected_landmarks:
            if i[0] in [17, 19, 21, 22, 24, 26, 51, 54, 57, 48]:
                current_coords.append(i[1][0, 0])
                current_coords.append(i[1][0, 1])
                continue
            else:
                if i[0] == 36:
                    leyel = i[1]
                if i[0] == 39:
                    leyer = i[1]
                if i[0] == 42:
                    reyel = i[1]
                if i[0] == 45:
                    reyer = i[1]
        leye_top, leye_bottom, reye_top, reye_bottom, leye_center, reye_center = fp.eyes_landmarks_estimation(frame)
        leyeH_proj = leye_bottom[1] - leye_top[1]
        leyeV_proj = leyer[0, 0] - leyel[0, 0]
        reyeH_proj = reye_bottom[1] - reye_top[1]
        reyeV_proj = reyer[0, 0] - reyel[0, 0]
        reyeR_proj = reyeH_proj / reyeV_proj
        leyeR_proj = leyeH_proj / leyeV_proj

        if ref_coords == []:
            ref_coords = current_coords
            GFeatures_frame = np.array(current_coords) - np.array(ref_coords)
            GFeatures_frame = np.append(np.append(GFeatures_frame, leyeR_proj), reyeR_proj)
        else:
            GFeatures_frame = np.array(current_coords) - np.array(ref_coords)
            GFeatures_frame = np.append(np.append(GFeatures_frame, leyeR_proj), reyeR_proj)
        GFeatures_collection.append(GFeatures_frame)
        GFvect= GFeatures_collection[1:]
        print(ref_coords)
        print(current_coords)
        print(GFeatures_frame)

    return GFvect