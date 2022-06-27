#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math
from random import randint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.filters import gabor_kernel
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny

from config import global_config
from scipy import misc, ndimage
CFG = global_config.cfg
SKIP_PLOTS = False

def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    # lines = probabilistic_hough_line(image, threshold=10, line_length=5,line_gap=3)


   
    
    # plot_figures(
    #     [
    #         {'name': 'binary_image_pre_morphological_process', 'image': image },
    #         {'name': 'closing', 'image': closing }
    #     ],
    #     skip=False
    # )
    output_image = np.zeros((image.shape[0], image.shape[1], 3))


    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(image, threshold=10, line_length=5, line_gap=3)
    for line in lines:
        # x_cords = [line[i][0] for i in range(len(line))]
        # y_cords = [line[i][1] for i in range(len(line))]
        # final_x_cords = [x for x in range(min(x_cords), max(x_cords) + 1, 1)]
        # final_y_cords = [int(round(y)) for y in np.interp(final_x_cords, x_cords, y_cords)]
        # for p in [p for p in zip(final_x_cords, final_y_cords) if p[1] >= 0 and p[1] < output_image.shape[0] and p[0] >= 0 and p[0] < output_image.shape[1]]:
        #     output_image.itemset((p[1], p[0]), 1)
        output_image = cv2.line(output_image, line[0], line[1], (255, 255, 255), 1)
    
    # fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].set_title('Input image')

    # ax[1].imshow(edges, cmap=plt.cm.gray)
    # ax[1].set_title('Canny edges')


    # ax[2].imshow(edges * 0)
    # for line in lines:
    #     output_image = cv2.line(output_image, line[0], line[1], (255,0,0), 3)
    #     p0, p1 = line
    #     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    # ax[2].set_xlim((0, image.shape[1]))
    # ax[2].set_ylim((image.shape[0], 0))
    # ax[2].set_title('Probabilistic Hough')
    # ax[3].imshow(output_image, cmap=plt.cm.gray)
    # ax[3].set_title('Lines')

    # for a in ax:
    #     a.set_axis_off()

    # plt.tight_layout()
    # plt.show()
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for x in range(len(output_image)):
        for y in range(len(output_image[x])):
            if tuple([int(v) for v in output_image[x][y]]) == (255, 255, 255):
                output[x][y] = 1
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    # closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    # abc = _connect_components_analysis(closing_2)
    # abc_1 = _connect_components_analysis(closing)
    # plt.figure('morpho')
    # plt.imshow(closing)
    # plt.savefig(f'kernel_size_{kernel_size}.png')
    
    # gauss_filter = ndimage.gaussian_filter(image, sigma=3)
    # plt.figure('gauss')
    # plt.imshow(gauss_filter)

    # new_kernel = np.real(gabor_kernel(10, theta=math.pi/2))
    # filtered = ndimage.convolve(image, new_kernel, mode='wrap')
    # plt.figure('gabor')
    # plt.imshow(filtered)
    # plt.show()

    return closing
    

def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # print(cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S))
    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([50, 100, 80]),
                           np.array([50, 100, 90]),
                           np.array([100, 50, 1020])]

    @staticmethod
    def _embedding_feats_dbscan_cluster(embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        features = StandardScaler().fit_transform(embedding_image_feats)
        #features = embedding_image_feats
        clust = OPTICS(min_samples=50, xi=.1, min_cluster_size=.2) #.05
       # import pdb; pdb.set_trace()
# Run the fit
        # plt.figure('embedding')
        # plt.plot(embedding_image_feats[:,0], embedding_image_feats[:,1], '.r', alpha=0.3)
        # plt.figure('features')
        # plt.plot(features[:,0], features[:,1], '.g', alpha=0.3)
        # plt.show()
        #exit()
        clust.fit(features)
        eps_1 = 0.35
        eps_2 = 0.1
        labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=eps_1)
        labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                           core_distances=clust.core_distances_,
                                           ordering=clust.ordering_, eps=eps_2)

        space = np.arange(len(features))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]

        plt.figure(figsize=(10, 7))
        G = gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(G[0, :])
        ax2 = plt.subplot(G[1, 0])
        ax3 = plt.subplot(G[1, 1])
        ax4 = plt.subplot(G[1, 2])

        # Reachability plot
        colors = ['g.', 'r.', 'b.', 'y.', 'c.']
        for klass, color in zip(range(0, 5), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            ax1.plot(Xk, Rk, color, alpha=0.3)
        ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
        ax1.plot(space, np.full_like(space, eps_2, dtype=float), 'k-', alpha=0.5)
        ax1.plot(space, np.full_like(space, eps_1, dtype=float), 'k-.', alpha=0.5)
        ax1.set_ylabel('Reachability (epsilon distance)')
        ax1.set_title('Reachability Plot')

        # OPTICS
        colors = ['g.', 'r.', 'b.', 'y.', 'c.']
        for klass, color in zip(range(0, 5), colors):
            Xk = features[clust.labels_ == klass]
            ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        ax2.plot(features[clust.labels_ == -1, 0], features[clust.labels_ == -1, 1], 'k+', alpha=0.1)
        ax2.set_title('Automatic Clustering\nOPTICS')

        # DBSCAN at 0.5
        colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
        for klass, color in zip(range(0, 6), colors):
            Xk = features[labels_050 == klass]
            ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
        ax3.plot(features[labels_050 == -1, 0], features[labels_050 == -1, 1], 'k+', alpha=0.1)
        ax3.set_title(f'Clustering at {eps_1} epsilon cut\nDBSCAN')

        # DBSCAN at 2.
        colors = ['g.', 'm.', 'y.', 'c.']
        for klass, color in zip(range(0, 4), colors):
            Xk = features[labels_200 == klass]
            ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        ax4.plot(features[labels_200 == -1, 0], features[labels_200 == -1, 1], 'k+', alpha=0.1)
        ax4.set_title(f'Clustering at {eps_2} epsilon cut\nDBSCAN')

        plt.tight_layout()
        plt.show()
        # db_1 = labels_050
        #exit(0)
        #db = DBSCAN(eps=eps_1, min_samples=50)
        db = DBSCAN(eps=CFG.POSTPROCESS.DBSCAN_EPS, min_samples=CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        db.fit(features)

        # try:
        #     features = StandardScaler().fit_transform(embedding_image_feats)
        #     #import pdb; pdb.set_trace()
        #     db.fit(features)
        # import pdb; pdb.set_trace()    


#         except Exception as err:
#             log.error(err)
#             ret = {
#                 'origin_features': None,
#                 'cluster_nums': 0,
#                 'db_labels': None,
#                 'unique_labels': None,
#                 'cluster_center': None
#             }
#             return ret
        db_labels = db.labels_
        #db_labels = labels_050
        core_samples_mask = np.zeros_like(db_labels, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise_ = list(db_labels).count(-1)
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        #cluster_centers = db.components_
        
        

# # Black removed and is used for noise instead.
#         unique_labels = set(db_labels)
#         colors = [plt.cm.Spectral(each)
#                   for each in np.linspace(0, 1, len(unique_labels))]
#         for k, col in zip(unique_labels, colors):
#             if k == -1:
#                 # Black used for noise.
#                 col = [0, 0, 0, 1]

#             class_member_mask = (db_labels == k)

#             xy = features[class_member_mask & core_samples_mask]
#             plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                      markeredgecolor='k', markersize=14)

#             xy = features[class_member_mask & ~core_samples_mask]
#             plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                      markeredgecolor='k', markersize=6)

#         plt.title('Estimated number of clusters: %d' % n_clusters_)
#         plt.savefig(f'eps_{CFG.POSTPROCESS.DBSCAN_EPS}_minS_{CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES}.png')
#         plt.show()
#         unique_labels = np.unique(db_labels)
        #exit(0)


        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels
            #'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """

        idx = np.where(binary_seg_ret == 255)
        #print(idx)
        #import pdb; pdb.set_trace()
        # plt.figure('imagen insta')
        # plt.plot(instance_seg_ret)
        # plt.show()
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        #import pdb; pdb.set_trace()

        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )
        # retorna un dicc con la wea marcada 2 valores
        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []
        
        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([0, 0, 0]),
                           np.array([0, 0, 0]),np.array([0, 0, 0]),np.array([0, 0, 0]),np.array([0, 0, 0]),np.array([0, 0, 0]),np.array([0, 0, 0]),np.array([0, 0, 0]),]

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=10)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        # lane line fit
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        for lane_index, coords in enumerate(lane_coords):
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
            elif data_source == 'beec_ccd':
                tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
            elif data_source == 'generic':
                source_image_height, source_image_width, _ = source_image.shape
                tmp_mask = np.zeros(shape=(source_image_height, source_image_width), dtype=np.uint8)
                tmp_mask[tuple((np.int_(coords[:, 1] * source_image_height / 256), np.int_(coords[:, 0] * source_image_width / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1]
        total_lines = {}
        line_string = 1
        #print("\n\n\n\n\n\n\n\n\n", len(src_lane_pts))
        #import pdb; pdb.set_trace()

        for index, single_lane_pts in enumerate(src_lane_pts):
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            elif data_source == 'beec_ccd':
                start_plot_y = 820
                end_plot_y = 1350
            elif data_source == 'generic':
                start_plot_y = 240
                end_plot_y = source_image_height
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            lines = []
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                idx_low = np.argmax(fake_diff_smaller_than_zero)
                idx_high = np.argmin(fake_diff_bigger_than_zero)

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue

                lane_color = self._color_map[index].tolist()
                lines.append([int(interpolation_src_pt_x), int(interpolation_src_pt_y)])
                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
            total_lines[str(line_string)] = lines
            x_p = []
            y_p = []
            for point_x, point_y in lines:
                x_p.append(point_x)
                y_p.append(point_y)
            total_lines[str(line_string)+'_copy']=[x_p, y_p]    
            line_string += 1
                
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
            'point_lines' : total_lines,
        }

        return ret


def plot_figures(figures, skip=False, save_files=False):
    if SKIP_PLOTS or skip: return
    if not isinstance(figures, list): figures = [figures]
    for figure in figures:
        name = figure['name'] if 'name' in figure else f'no-name-{randint(1, 100)}'
        plt.figure(name)
        plt.imshow(figure['image'], **(figure['kwargs'] if 'kwargs' in figure else {}))
        if save_files: cv2.imwrite(name, figure['image'])
    plt.show()
