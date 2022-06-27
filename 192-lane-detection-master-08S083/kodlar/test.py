import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

class Test():

  def __init__(self):
      self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

      self.net = lanenet.LaneNet(phase='test', net_flag='vgg')
      self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='lanenet_model')

      self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

      self.saver = tf.train.Saver()

    # Set sess configuration
      self.sess_config = tf.ConfigProto()
      self.sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
      self.sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
      self.sess_config.gpu_options.allocator_type = 'BFC'

      self.sess = tf.Session(config=self.sess_config)
      self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()
  
  
  def image_process(self, image_path, weights_path):
      
      self.t_start = time.time()
      self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)

      self.image_vis = self.image
      self.image = cv2.resize(self.image, (512, 256), interpolation=cv2.INTER_LINEAR)
      self.image = self.image / 127.5 - 1.0
      log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - self.t_start))

      with self.sess.as_default():

          self.saver.restore(sess=self.sess, save_path=weights_path)

          self.t_start = time.time()
          self.binary_seg_image, self.instance_seg_image = self.sess.run(
                                                  [self.binary_seg_ret, self.instance_seg_ret],
                                                  feed_dict={self.input_tensor: [self.image]}
          )
          
          self.t_cost = time.time() - self.t_start
          log.info('Single imgae inference cost time: {:.5f}s'.format(self.t_cost))

          self.postprocess_result = self.postprocessor.postprocess(
              binary_seg_result=self.binary_seg_image[0],
              instance_seg_result=self.instance_seg_image[0],
              source_image=self.image_vis,
              min_area_threshold=CFG.POSTPROCESS.MIN_AREA_THRESHOLD,
              data_source='generic'
              )
          self.mask_image = self.postprocess_result['mask_image']
        #print(postprocess_result)
          for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
              self.instance_seg_image[0][:, :, i] = minmax_scale(self.instance_seg_image[0][:, :, i])
          self.embedding_image = np.array(self.instance_seg_image[0], np.uint8)
          lanenet_postprocess.plot_figures([
            {'name': 'mask_image', 'image': self.mask_image[:, :, (2, 1, 0)]},
            {'name': 'src_image', 'image': self.image_vis[:, :, (2, 1, 0)]},
            {'name': 'instance_image', 'image': self.embedding_image[:, :, (2, 1, 0)]},
            {'name': 'binary_image', 'image': self.binary_seg_image[0] * 255, 'kwargs': {'cmap': 'gray'} },
        ], skip=False, save_files=False)
  
  
  def close_sess(self):
      
      self.sess.close()



weights_path = './model/pesos_finales/tusimple_lanenet_vgg_2020-06-20-22-26-00.ckpt-80001'

images = Test()
images.image_process('./data/tusimple_test_image/frame0.png', weights_path)
images.image_process('./data/tusimple_test_image/frame30.png', weights_path)
images.image_process('./data/tusimple_test_image/frame50.png', weights_path)

images.close_sess()
