from FrameIterator_test import get_unique_paths, test_frame_iterator
from GFExtractor import GFeatures
from LBPFExtractor import LBPFeatures
import numpy as np

unique_paths = get_unique_paths('pics')
print(unique_paths)

print('LBP features extraction ...')

LBPF = []
for un in unique_paths:
    files = test_frame_iterator(un)
    for file in files:
        print(un+file)
    LBPF_ = LBPFeatures(un, 'face_landmarks/shape_predictor_68_face_landmarks.dat')

    for _ in LBPF_:
        LBPF.append(_)

print(len(LBPF))
np.savetxt("features_vectors/train/LBPF.csv", LBPF, delimiter=",")

print('Geometric features extraction ...')

GF = []
for un in unique_paths:
    files = test_frame_iterator(un)
    for file in files:
        print(un+file)
    GF_ = GFeatures(un, 'face_landmarks/shape_predictor_68_face_landmarks.dat')
    for _ in GF_:
        GF.append(_)
print(len(GF))
np.savetxt("features_vectors/train/GF.csv", GF, delimiter=",")