#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import glob
import imageio
import numpy as np

# ---用于平面拟合
import functools
import scipy.optimize

# from scipy import signal
# import matplotlib.pyplot as plt

# from utils import logger

import logging

import datetime

from tof_utils import c_tof_depth2voxel, c_tof_template_match


class Detector(object):
    def __init__(self, pallet_type):
        self.oversize_detection_params = {
            ###########################################################################################################
            # ------------------ 相机标定参数
            'DEPTH_FACTOR': 0.3333,
            # -----------------
            'T_DIST_MIN': 450,      # mm，使用超板检测算法的最小距离，小于该距离时，不能捕获到完整的卡板脚，故无法使用该算法
            'T_DIST_MAX': 1400,     # ToF相机的最大量程
            # 'CAM_K': [[518.8579, 0,        240],
            #           [0,        518.8579, 320],
            #           [0,        0,          1]],  # 已将图片向右旋转90度, H, W = 640, 480
            'CAM_K': [[500.649170, 0, 245.441635],
                      [0, 500.234070, 318.240936],
                      [0, 0, 1]],

            ###########################################################################################################
            #
            # ------------------ PALLET LEG INFO in depth
            # 卡板脚的宽度范围: 长度 17 正负5, 单位 pixel
            'PALLET_LEG_WIDTH_MIN': 12,
            'PALLET_LEG_WIDTH_MAX': 60,
            'PALLET_BODY_HEIGHT_MAX': 15,  # 卡板横木的厚度，最多向上查找PALLET_BODY_HEIGHT_MAX行
            #
            # ------------------ PALLET LEG INFO in 3D grid
            # 'PALLET_LEG_MASK': np.ones((4, 10, 2), dtype=np.uint8),  # wooden pallet leg
            'PALLET_LEG_MASK': np.ones((5, 10, 1), dtype=np.uint8),  # voxel
            # 'PALLET_LEG_MASK': np.ones((8, 10, 3), dtype=np.uint8),  # voxel
            'GRID_UNIT': 10,                # mm
            'GRID_OFFSET': [500, 200, 0],   # [x, y, z] mm
            'GRID_OFFSET_FLOOR': 440,       # mm, floor = Y + 300
            'GRID_SIZE': [80, 120, 140],    # (X, Y, Z) voxel
            'GRID_UNIT_CELEBRATE': 11.6,    # mm, 实际上每个体素对应的物理尺寸

            ###########################################################################################################
            # ----------------- 程序员调参
            #
            'T_GRADIENT': 7,  # 对水平线进行聚类时的阈值，需实验确定与验证，[]
            'T_NUMBER_OF_VALID_POINTS_PER_LINE': 10,  # 每一行中有效点的最少数目[]
            'T_LINE_MIN_LENGTH': 10,  # 有效分割线段的最小长度，测量卡板脚的宽度在深度相机中的最远处时所占的像素个数，据此确认

            ###########################################################################################################
            # ----------------- 后续可能会舍弃的参数
            'T_PLANE_ERROR': 0.04,  # 判断一个点是否属于某一平面的误差阈值，取值范围: [0.04, 0.25]
            # --- 同时增大或减小ground_top与ground_bottom
            # 由于在点云坐标系中，上下颠倒，故ground_bottom >ground_top
            'ground_top': 280,
            'ground_bottom': 380,
            'COLUMN': -1,  # 268卡板脚的对应位置在Depth中的列数
            # 指定已知区域，提供四边形（梯形）的四个顶点的坐标（x, y）， 依次为[左上，右上，左下，右下]
            'PLANE': [(190, 540), (280, 540), (140, 620), (280, 620)],
            'PLANE_SEARCH_AREA': {'min_x': 50, 'max_x': 430, 'min_y': 400, 'max_y': 590},  # min x, max x, min y, max y
        }

        self.logger = self.set_logger()
        self.set_pallet_type(pallet_type)

        # ---- Viz
        self.mask_color_dict = {
            'v_line': [0, 255, 0],  # 所有分割线段
            'v_line_pallet_leg': [255, 255, 0],  # 卡板脚
            'v_line_pallet_body': [255, 255, 0],  # 卡板横木
            'point_goods_edge': [199, 21, 133],  # 货物边缘点
            'plane_initial_area': [199, 21, 133],  # 平面初始区域
            'plane_search_area': [125, 255, 0],  # 平面搜索区域
            'plane_estimated_area': [200, 200, 0],  # 平面估计结果中的全部平面区域
        }

        self.color_dict = {
            'h_line_pallet': [255, 0, 0],  # 卡板外侧所在位置的垂直示意线
            'h_line_overstep': [0, 0, 139],  # 超板货物所在位置的垂直示意线
            'point_overstep': [255, 255, 255],  # 超板货物所在位置，仅显示超板为多的一个点
        }

    def set_pallet_type(self, pallet_type):
        if pallet_type is 'WOODEN':
            self.oversize_detection_params['PALLET_LEG_MASK'] = self.get_wooden_mask()
        elif pallet_type is 'PLASTIC':
            self.oversize_detection_params['PALLET_LEG_MASK'] = self.get_plastic_mask()
        else:
            print("Pallet type error! Please select type from: ['WOODEN', 'PLASTIC].")

    @staticmethod
    def get_plastic_mask():
        top = 14
        bottom = 6
        height = 15
        dx = (top - bottom) / 2   # width
        dz = 2  # depth
        width = top
        mask = np.zeros((width, height, 4), dtype=np.uint8)
        for i_line in range(height):
            x_bar = int(round(1.0 * dx * i_line / height))
            z_bar = int(round(1.0 * dz * i_line / height))
            # print(x_bar, z_bar)
            mask[x_bar:width-x_bar, i_line, z_bar:z_bar+2] = 1
        # print(mask.shape)
        # print(mask)
        return mask

    @staticmethod
    def get_wooden_mask():
        return np.ones((5, 10, 1), dtype=np.uint8)

    def run(self, depth_img, plyname=None):
        #
        depth_img = depth_img * self.oversize_detection_params['DEPTH_FACTOR']  # numpy.float64
        #
        unit = self.oversize_detection_params['GRID_UNIT']
        o_xyz = self.oversize_detection_params['GRID_OFFSET']  # mm
        grid_size = self.oversize_detection_params['GRID_SIZE']

        cam_k = np.asarray(self.oversize_detection_params['CAM_K']).astype(np.float32)
        arr_mask = self.oversize_detection_params['PALLET_LEG_MASK']
        # print('arr_mask', arr_mask.shape)

        # 0.03s
        voxel_count, index_depth2grid, index_grid2depth = c_tof_depth2voxel(depth_img,
                                                                            unit=unit,
                                                                            gridsize=grid_size,
                                                                            offset=o_xyz,
                                                                            cam_k=cam_k)

        # ------ 货物必须与卡板正面是对齐的。
        # b_is_valid = Detector.valid_depth_range(voxel_count, self.oversize_detection_params)  # 500 mm
        # if b_is_valid is False:
            # return

        # -----------------------------------
        floor_height_min = int(self.oversize_detection_params['GRID_OFFSET_FLOOR'] / unit)  # voxels
        pallet_dist, floor_height = Detector.height_map(voxel_count, floor_height_min)
        # print('pallet_dist: ', pallet_dist, 'floor_height: ', floor_height)  # TODO logger

        _t_unit = self.oversize_detection_params['T_DIST_MIN'] / self.oversize_detection_params['GRID_UNIT']
        if pallet_dist < _t_unit:
            print('-------- The pallet distance {}(< {}), is too close. ---------'.format(pallet_dist, _t_unit))
            result_dict = {
                'state': 'FAIL',
                'info': 'The pallet distance {}(< {}), is too close.'.format(pallet_dist, _t_unit)
            }
            return -1, -1, result_dict

        # --- Step 1, 在深度图指定区域的素点进行平面拟合，该区域为先验的地面区域
        # 然后在扩大的区域内，依据点到平面的距离，判断对应像素是否属于地面，从而获取所有属于地面的像素点
        # t1 = datetime.datetime.now()
        # points_of_plane_index_array = find_ground(depth_img, oversize_detection_params)
        # print('Timed used in find_ground: {}'.format(datetime.datetime.now() - t1))
        # position_xyz = point_grid[points_of_plane_index_array]
        # position_xyz = point_grid[points_of_plane_index_array[:, 0], points_of_plane_index_array[:, 1], :]
        # ---- get Z of the ground voxels
        # print('ground voxel: Z', np.mean(position_xyz[:, 2]))

        # --- Step 2
        # print('x, y, z', np.max(position_xyz[:,0]), np.max(position_xyz[:,1]), np.max(position_xyz[:,2]))
        # t2 = datetime.datetime.now()
        # match_pallet_leg_from_voxels(voxel_binary, position_xyz)
        # print('Python: Timed used in match_pallet_leg_from_voxels: {}'.format(datetime.datetime.now() - t2))

        tx, ty, tz = c_tof_template_match(voxel_count,
                                          floor_y=floor_height,
                                          mask=arr_mask)  # 0.02s
        # print('Detected pallet leg position:', tx, ty, tz)                # TODO: Logger
        # print('Find Pallet Leg at: ', index_grid2depth[tx, ty, tz, :])    # TODO: Logger

        self.oversize_detection_params['COLUMN'] = index_grid2depth[tx, ty, tz, 1]  # index_pallet_leg
        # logger.debug("oversize_detection_params['COLUMN']:{}".format(oversize_detection_params['COLUMN']))

        # --- Step 3,
        results = Detector.find_edges_of_pallet_and_goods(depth_img,
                                                          self.oversize_detection_params,
                                                          self.mask_color_dict)
        # --- Step 4
        if results['state'] == 'OK':
            x_pixel = results['pointx_pallet'] - results['pointx_goods']
            x_unit_pallet = index_depth2grid[results['pointy_pallet'], results['pointx_pallet'], 0]  # [h, w, 3(x y z)]
            x_unit_goods = index_depth2grid[results['pointy_goods'], results['pointx_goods'], 0]
            # print("pallet corner in grid(x y z):", point_grid[results['pointy_pallet'], results['pointx_pallet'], :])
            x_unit = x_unit_pallet - x_unit_goods
            # x_mm = x_unit * self.oversize_detection_params['GRID_UNIT']
            x_mm = x_unit * self.oversize_detection_params['GRID_UNIT_CELEBRATE']
            # print("Over size detection:{} pixels, {} unit, {} mm".format(x_pixel, x_unit, x_mm))  # TODO: logger
            # ---- 将不超板的情况，全部设为超板0
            x_pixel = 0 if x_pixel < 0 else x_pixel
            x_mm = 0 if x_mm < 0 else x_mm

        else:
            print("Over size detection failed.")
            return -2, -2, results

        # ----- 检测结果可视化
        if plyname is not None:
            pt = list()
            pt.append(index_depth2grid[results['pointy_pallet'], results['pointx_pallet'], :])
            pt.append(index_depth2grid[results['pointy_goods'], results['pointx_goods'], :])
            Detector.show_pallet_leg(voxel_count, tx, ty, tz, plyname, pt=pt, floor=floor_height)

        # result_dict = {
        #     'state': 'OK',
        #     'info': 'OK',
        #     # 'point_overstep': (overstep_points_x, overstep_points_y),
        #     # 'point_overstep': (overstep_points_x, overstep_points_y),
        #     'pointx_goods': overstep_points_x,
        #     'pointy_goods': overstep_points_y,
        #     'line_goods_array': line_goods_array,
        #     'line_pallet_array': line_pallet_array,
        #     'line_pallet_leg_list': line_pallet_leg_list,
        #     'line_pallet_body_list': line_pallet_body_list,
        #     # 'pointx_pallet_edge': pointx_pallet_edge,
        #     'pointx_pallet': pointx_pallet_edge,
        #     'pointy_pallet': pointy_pallet_body_top_line,
        #     'pointy_pallet_leg_top_line': pointy_pallet_leg_top_line,
        #     # 'pointy_pallet_body_top_line:': pointy_pallet_body_top_line,
        #     'mask': mask_edges
        # }

        return x_pixel, x_mm, results

    def viz_result(self, depth, rgb, results, vizname):
        # ----- 检测结果可视化
        if vizname is not None and results['state'] == 'OK' and results['mask'] is not None:
            # ---- Mark the pixels without depth value in red
            img_viz = Detector.mark_missing_pixels(rgb, depth, np.array([100, 0, 0]))
            # --- 显示线段聚类的结果
            img_viz = Detector.viz_line_segment(img_viz, line_mask=results['mask'])
            # --- 显示卡板的边缘
            img_viz = Detector.viz_edge_points(img_viz, results['pointx_pallet'], self.color_dict['h_line_pallet'])
            # --- 显示超板货物的边缘
            # img_viz = viz_edge_points(img_viz, results['point_overstep'][0], color_dict['h_line_overstep'])
            img_viz = Detector.viz_edge_points(img_viz, results['pointx_goods'], self.color_dict['h_line_overstep'])
            # --- 显示超板货物的边缘中最突出的那一个点
            # img_viz = viz_overstep_point(img_viz, results['point_overstep'], color_dict['point_overstep'])
            img_viz = Detector.viz_overstep_point(img_viz,
                                                  (results['pointx_goods'], results['pointy_goods']),
                                                  self.color_dict['point_overstep'])
            # --- 显示卡板的左上角
            img_viz = Detector.viz_overstep_point(img_viz,
                                                  (results['pointx_pallet'], results['pointy_pallet']),
                                                  [255, 0, 255])
            imageio.imwrite(vizname, img_viz)
            #cv2.imshow('img_viz', img_viz)

    @staticmethod
    def valid_depth_range_old(depth_img, t):
        mean_depth = np.mean(depth_img[280:360, 200:280])  # 实验选取范围
        if mean_depth < t:
            print('Too close, mean_depth < {}: ------------------------'.format(t))
            return False
        else:
            return True

    @staticmethod
    def height_map(grid, floor_height_min):
        # x: width
        # y: height
        # z: depth
        # ---- get size
        size = grid.shape
        binary_grid = np.zeros(size, dtype=np.uint8)
        binary_grid[grid > 0] = 1

        # -------------------------------------------------------------------------
        # ---- Sum of array elements over a given axis.
        height_map_xz = np.sum(binary_grid, axis=1, initial=0)  # over Y, height
        binary_occupy_map = np.zeros(height_map_xz.shape, dtype=np.uint8)
        binary_occupy_map[height_map_xz > 2] = 1
        height_map_z = np.sum(binary_occupy_map, axis=0, initial=0)  # over X, width
        pallet_dist = np.argmax(height_map_z)  # get Z

        # -------------------------------------------------------------------------
        height_map_xy = np.sum(binary_grid, axis=2, initial=0)  # over Z, depth
        # binary_occupy_map2 = np.zeros(height_map_xy.shape, dtype=np.uint8)
        # binary_occupy_map2[height_map_xy > 3] = 1
        height_map_xy[height_map_xy < 3] = 0
        height_map_y = np.sum(height_map_xy, axis=0, initial=0)  # over X, width
        # --- y should be restricted to a specific area.
        # floor_height = np.argmax(height_map_y)  # get y
        restrictedy = height_map_y[floor_height_min:]
        floor_height = np.argmax(restrictedy) + floor_height_min  # get y

        return pallet_dist, floor_height

    @staticmethod
    def show_pallet_leg(voxel_val, tx, ty, tz, ply_filename, pt=None, floor=None):
        # ---- get size
        size = voxel_val.shape
        _x, _y, _z = Detector.get_xyz(size)
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()

        ply_data_rgb = np.zeros(size + (3,), dtype=np.int32)

        # -------------------------set color------------------------
        # --- Set pallet leg to color[255, 0, 0]
        ply_data_rgb[tx - 2:tx + 2, ty - 5:ty + 5, tz - 1:tz + 1, 0] = 255
        ply_data_rgb[tx - 2:tx + 2, ty - 5:ty + 5, tz - 1:tz + 1, 1] = 0
        ply_data_rgb[tx - 2:tx + 2, ty - 5:ty + 5, tz - 1:tz + 1, 2] = 0

        # ----在ply上发，显示指定位置的点
        if pt is not None:  # pt list of tuple (x, y, z)
            for _pt in pt:
                _ptx, _pty, _ptz = _pt
                ply_data_rgb[_ptx, _pty, _ptz, 0] = 0
                ply_data_rgb[_ptx, _pty, _ptz, 1] = 255
                ply_data_rgb[_ptx, _pty, _ptz, 2] = 255

        # -----------------------------------------------------------------
        # print('Set floor {} to color[0,0,255].'.format(floor))
        fs = 1
        if floor is not None:  #
            ply_data_rgb[:, floor-fs:floor+fs, :, 0] = 0
            ply_data_rgb[:, floor-fs:floor+fs, :, 1] = 0
            ply_data_rgb[:, floor-fs:floor+fs, :, 2] = 255

        # -----------------------------------------------------------------
        r = ply_data_rgb[:, :, :, 0].flatten()
        g = ply_data_rgb[:, :, :, 1].flatten()
        b = ply_data_rgb[:, :, :, 2].flatten()
        ply_data_grid = zip(_x, _y, _z, r, g, b, voxel_val.flatten())
        ply_data_grid = list(ply_data_grid)

        ply_data = []
        for i_idx in range(len(ply_data_grid)):
            if ply_data_grid[i_idx][6] > 3:  # 0 is empty, 1, only 1 point
                ply_data.append(ply_data_grid[i_idx])

        if len(ply_data) == 0:
            print('From _depth_voxel2ply(): NO valid data. {}'.format(ply_filename))
            return

        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'property int label\n' \
                   'end_header' % len(ply_data)
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d %d", header=ply_head, comments='')
        print('Saved-->{}'.format(ply_filename))

    @staticmethod
    def set_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        # logger.setLevel(level=logging.INFO)

        # --- 创建一个FileHandler，并对输出消息的格式进行设置，将其添加到logger，然后将日志写入到指定的文件中
        # file_handler = logging.FileHandler("log.txt")
        # file_handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

        # --- logger中添加StreamHandler，可以将日志输出到屏幕上
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        return logger

    def read_rgb(self, rgb_filename):  # 0.01s
        r"""Read a RGB image with size H x W x channel
        """
        # rgb = misc.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        rgb = imageio.imread(rgb_filename)
        # rgb = misc.imresize(rgb, (img_h, img_w))  # (H, W, 3)
        # rgb = np.rollaxis(rgb, 2, 0)  # (H, W, 3)-->(3, H, W)
        self.logger.debug('RGB shape: {}'.format(rgb.shape))
        return rgb

    # @staticmethod
    def read_depth(self, depth_filename):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        depth = imageio.imread(depth_filename)  # <class 'numpy.uint16'>
        # depth = (misc.imread(depth_filename) / 1000.0 ) * 0.3333  # numpy.float64
        # depth = misc.imresize(depth, (img_h, img_w))  # numpy.uint8
        # assert depth.shape == (img_h, img_w), 'incorrect default size'
        # print('Max depth :', np.amax(depth))
        self.logger.debug('Depth shape: {}'.format(depth.shape))
        return depth

    @staticmethod
    def get_xyz(size):
        _x = np.zeros(size, dtype=np.int32)
        _y = np.zeros(size, dtype=np.int32)
        _z = np.zeros(size, dtype=np.int32)

        for i_h in range(size[0]):  # x, y, z
            _x[i_h, :, :] = i_h  # x, left-right flip
        for i_w in range(size[1]):
            _y[:, i_w, :] = i_w  # y, up-down flip
        for i_d in range(size[2]):
            _z[:, :, i_d] = i_d  # z, front-back flip
        return _x, _y, _z

    # 将深度值缺失对应像素点标记其颜色为color
    @staticmethod
    def mark_missing_pixels(rgb, depth, color=np.array([255, 0, 0])):
        img_h, img_w, _ = rgb.shape
        str_info = 'RGB size:{}x{}, Depth size:{}x{}'.format(img_h, img_w, depth.shape[0], depth.shape[1])
        assert depth.shape == (img_h, img_w), 'RGB image and Depth shape matching error. ' + str_info
        mask = np.zeros(rgb.shape, dtype=np.uint8)
        mask[depth == 0.0, :] = color
        mask_rgb = cv2.addWeighted(rgb, 1, mask, 0.6, 0)
        return mask_rgb

    @staticmethod
    def get_point_cloud(params, depth, rgb=None):
        """ Get point cloud from depth(H, W). Return list of [x, y, z]
        or Get color point cloud from RGB-D, rgb (H, W, 3) and depth(H, W). Return list of [x, y, z, r, g, b]
        """  #
        cn = 3 if rgb is None else 6
        cam_k = params['CAM_K']
        image_h, image_w = depth.shape
        # ---- Get point in camera coordinate
        gx, gy = np.meshgrid(range(image_w), range(image_h))
        point_cam = np.zeros((image_h, image_w, cn), dtype=np.float32)
        point_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
        point_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
        point_cam[:, :, 2] = depth  # z
        if rgb is not None:
            point_cam[:, :, 3] = rgb[:, :, 0]
            point_cam[:, :, 4] = rgb[:, :, 1]
            point_cam[:, :, 5] = rgb[:, :, 2]
        return point_cam

    @staticmethod
    def save_point_cloud(point_clouds, ply_filename):
        """ Save point clouds (x, y, z) or color point coluds (x, y, z, r, g, b) 
        """  #
        # Shape of point_clouds is H x W x n
        points_number = point_clouds.shape[0] * point_clouds.shape[1]
        n = point_clouds.shape[2]
        if n == 3:  # (x, y, z)
            ply_head = 'ply\n' \
                       'format ascii 1.0\n' \
                       'element vertex %d\n' \
                       'property float x\n' \
                       'property float y\n' \
                       'property float z\n' \
                       'end_header' % points_number
            # ply_filename = filename + '_Depth2CPL.ply'
            ply_data = point_clouds.reshape((-1, n))
            np.savetxt(ply_filename, ply_data, fmt="%.2f %.2f %.2f", header=ply_head, comments='')
        elif n == 6:
            ply_head = 'ply\n' \
                       'format ascii 1.0\n' \
                       'element vertex %d\n' \
                       'property float x\n' \
                       'property float y\n' \
                       'property float z\n' \
                       'property uchar red\n' \
                       'property uchar green\n' \
                       'property uchar blue\n' \
                       'end_header' % points_number
            # ply_filename = filename + '_RGBD2CPL.ply'
            ply_data = point_clouds.reshape((-1, n))
            np.savetxt(ply_filename, ply_data, fmt="%.2f %.2f %.2f %d %d %d", header=ply_head, comments='')
        # logger.info('Saved-->{}'.format(ply_filename))

    @staticmethod
    def get_initial_ground_index_tuple(point_cloud, params):
        # ground_top = 280
        # ground_bottom = 380
        ground_top = params['ground_top']
        ground_bottom = params['ground_bottom']
        _idx = np.where(np.logical_and(point_cloud[:, :, 1] > ground_top, point_cloud[:, :, 1] < ground_bottom))
        return _idx

    @staticmethod
    def find_ground(depth, params):
        # --- depth 转为点云
        point_cloud = Detector.get_point_cloud(params, depth)
        # print('point_cloud.shape:', point_cloud.shape)

        # index_array = get_initial_ground_index_array(depth, params)
        # points = point_cloud[index_array[:, 0], index_array[:, 1], :] / 1000.0

        index_tuple = Detector.get_initial_ground_index_tuple(point_cloud, params)

        # --- resample to reduce the number of optimized points
        # print(type(index_tuple), len(index_tuple), type(index_tuple[0]), index_tuple[0].shape)
        a = index_tuple[0].size
        num_re = 1000  # TODO if a <1000, if a< 20??, if a==0
        inds = np.random.choice(a, num_re, replace=False)

        index_tuple2 = (index_tuple[0][inds], index_tuple[1][inds])

        # index_initial_points = index_tuple
        index_initial_points = index_tuple2

        # ------
        points = point_cloud[index_initial_points[0], index_initial_points[1], :] / 1000.0

        # logger.debug("In 'find_ground': points.shape is {}".format(points.shape))  # (1473, 3)

        # --- 进行平面拟合
        fun = functools.partial(Detector.error, points=points)
        # TODO: ToF 固定安装之后，进行一次测试，取拟合后的结果作为初始值，以减少后期拟合的迭代次数
        # params0 = np.array([0, 0, 0])
        params0 = np.array([-0.2, -4, 2])  # initial guess
        res = scipy.optimize.minimize(fun, params0)  # 这一步很耗时

        a = res.x[0]
        b = res.x[1]
        c = res.x[2]

        indexes_all = Detector.get_index(depth.shape[0], depth.shape[1])
        psa = params['PLANE_SEARCH_AREA']
        indexes = indexes_all[psa['min_y']:psa['max_y'], psa['min_x']:psa['max_x'], :]  # search area
        indexes = np.reshape(indexes, (-1, 2))
        indexes = indexes.tolist()
        plane_points_list = indexes

        ground_points_list = list()
        d_point2plane_list = list()
        for p in plane_points_list:
            if depth[p[0], p[1]] == 0.0:  # --- Filter out pixels valued zero in depth
                continue
            if point_cloud[p[0], p[1], 1] < 200:  # --- Filter out points at a very high position
                continue
            # ppt = (point_cloud[p[0], p[1], 0], point_cloud[p[0], p[1], 1], point_cloud[p[0], p[1], 2])/1000.0
            ppt = (
                point_cloud[p[0], p[1], 0] / 1000.0, point_cloud[p[0], p[1], 1] / 1000.0,
                point_cloud[p[0], p[1], 2] / 1000.0)
            # print('ppt', ppt)
            d = Detector.distance_of_point_to_plane([a, b, c], ppt)
            d_point2plane_list.append(d)
            if d < params['T_PLANE_ERROR']:
                # print(p)
                ground_points_list.append(p)
            # print(d)
        # logger.debug('find {} ground points'.format(len(ground_points_list)))
        # logger.debug('Max distance of points to the plane:{}'.format(max(d_point2plane_list)))

        # points = [(1.1,2.1,8.1),
        #           (3.2,4.2,8.0),
        #           (5.3,1.3,8.2),
        #           (3.4,2.4,8.3),
        #           (1.5,4.5,8.0)]

        # xs, ys, zs = zip(*points)
        # return index_array
        # print(np.asarray(ground_points_list).shape)
        return np.asarray(ground_points_list)

    @staticmethod
    def find_edges_of_pallet_and_goods(depth, params, viz_color_dict=None):
        # -----------------------------------------------------------------------
        # PALLET_BODY_HEIGHT_MAX = 30  # 最多向上查找PALLET_BODY_HEIGHT_MAX行
        # ----- 初始化
        # line_segment_list
        dep_h, dep_w = depth.shape
        mask_edges = None
        line_pallet_leg_list = list()  # list of tuple, each tuple is (row, col_start, col_end, col_length)
        line_seg_list = list()  # list of tuple, each tuple is (row, col_start, col_end, col_length)
        if viz_color_dict is not None and len(viz_color_dict) > 0:
            mask_edges = np.zeros((dep_h, dep_w, 3), dtype=np.uint8)

        # --------------------------------------------------------------------------------------------------------
        # 逐行遍历Depth,从每一行的多个线段中，找到index包含固定列数的线段
        for idx_h in range(dep_h):
            # print('processing line index: idx_h=', idx_h)
            # depth_line_left = depth_img[idx_h, :]
            #
            # ------ 提取单行数据中的多个线段
            segment_list = Detector.find_line_segment(depth, idx_h, params)
            # print('Line {} has {} line_segments.'.format(idx_h, len(segment_list)))
            #
            # ------ 查找index包含固定列数的线段, 依据卡板脚的位置调整搜索范围
            line_seg = None
            line_pallet_leg = None
            for ls in segment_list:  # list of tuple(index_y, start_x, end_x, length)
                # if ls[1] < params['COLUMN'] < ls[2] or ls[1] < params['COLUMN'] + 80 < ls[2]:  # 卡板脚的所在列的位置范围
                # ----- line_seg_list: pallet leg, pallet body, goods
                # 卡板脚的所在列的位置范围pallet leg length, 超板 or 不超板
                if ls[1] < params['COLUMN'] < ls[2] \
                        or ls[1] < params['COLUMN'] + 80 < ls[2]:
                    # if params['COLUMN'] < ls[2]:  # 卡板脚的所在列的位置范围
                    # if ls[1] < params['COLUMN'] :  # 卡板脚的所在列的位置范围
                    line_seg = ls
                    # line_seg_list.append((idx_h, ls[0], ls[1], ls[2]))
                    line_seg_list.append((ls[0], ls[1], ls[2], ls[3]))
                    # line_seg_list.extend(ls)
                    # 查找卡板脚
                    # 查找卡板脚的Y轴范围最小值，放入 params
                    # ----- pallet leg
                    if idx_h > 350 \
                            and ls[1] < params['COLUMN'] < ls[2] \
                            and params['PALLET_LEG_WIDTH_MIN'] < ls[3] < params['PALLET_LEG_WIDTH_MAX']:
                        # line_pallet_leg = ls
                        # line_pallet_leg_list.append((idx_h, ls[0], ls[1], ls[2]))
                        line_pallet_leg_list.append((ls[0], ls[1], ls[2], ls[3]))
                        # line_pallet_leg_list.extend(ls)
            if 'v_line' in viz_color_dict and line_seg:
                mask_edges[idx_h, line_seg[1]:line_seg[2], :] = viz_color_dict['v_line']
            if 'v_line_pallet_leg' in viz_color_dict and line_pallet_leg:
                mask_edges[idx_h, line_pallet_leg[1]:line_pallet_leg[2], :] = viz_color_dict['v_line_pallet_leg']
        # -----------------------------------------------------------------------
        # ------ 在找到的卡板脚的上方，寻找卡板的木头盖板，然后拼接为完整卡板
        if len(line_pallet_leg_list) == 0:
            result_dict = {
                'state': 'FAIL',
                'info': "Failed to find the pallet leg.",
                'mask': mask_edges,
                #
                'pointx_goods': 0,
                'pointy_goods': 0,
                'pointx_pallet': 0,
                'pointy_pallet': 0,
            }
            # logger.warning("Failed to find the pallet leg.")  #  失败情况处理
            return result_dict

        # n * 4, (index_y, start_x, end_x, length_x)
        line_pallet_leg_array = np.asarray(line_pallet_leg_list)
        line_seg_array = np.asarray(line_seg_list)
        # ------ 沿着Y方向，向上方搜索一小段距离，查找属于卡板侧面的边界点
        # min_col = np.min(line_pallet_leg_array[:, 0])  # line_pallet_leg_array[0, 0]
        # max_col = np.max(line_pallet_leg_array[:, 0])  # line_pallet_leg_array[-1, 0]
        # logger.debug('line_pallet_leg_array.shape:{}'.format(line_pallet_leg_array.shape))
        # min_col = line_pallet_leg_array[0, 0]
        # max_col = line_pallet_leg_array[-1, 0]
        _leg_col = line_pallet_leg_array[:, 0]
        min_col = np.amin(_leg_col)
        max_col = np.amax(_leg_col)
        # logger.debug('min_col: {},  max_col: {}'.format(min_col, max_col))
        # print('min_col: {},  max_col: {}'.format(min_col, max_col))
        # ----- 找到min_col在line_seg_list中的位置,即卡板脚的top_line对应在整体线段行中的位置
        _i, = np.where(line_seg_array[:, 0] == min_col)  #  完善情况处理
        # if _i.size != 1:  #
        #     logger.warning("Find 'pointy_pallet_leg_top_line' Error. '_i'={}".format(_i))
        index_line_pallet_leg_top_line = _i[0]
        pointy_pallet_leg_top_line = line_seg_array[_i[0], 0]
        # pointy_pallet_leg_top_line = line_seg_array[index_line_pallet_leg_top_line, 0]

        cur_y, cur_x = line_pallet_leg_array[0, 0:2]
        # print('pointy_pallet_leg_top_line', pointy_pallet_leg_top_line, 'cur_y, cur_x', cur_y, cur_x)

        # 在分割得到的线段中，依次查找属于卡板body的线段。
        #  可能会遇到问题：当一行存在多条线段时，是否适用，需要check
        line_pallet_body_list = list()
        c = 0
        while c < params['PALLET_BODY_HEIGHT_MAX']:  # 最多向上查找PALLET_BODY_HEIGHT_MAX行
            c = c + 1
            # lsa = line_seg_array[i-c][0]
            lsa = line_seg_array[index_line_pallet_leg_top_line - c]
            _y, _x = lsa[0:2]
            # print('c: ', c, ' lsa:', lsa, ' _y, _x ', _y, _x)
            # print('dety: ', abs(_y - cur_y), ' detx: ', abs(_x - cur_x))
            if abs(_y - cur_y) < 1.1 and abs(_x - cur_x) < 3:  # 超参数
                # 将当前点加入到卡板的边界点中
                line_pallet_body_list.append((_y, _x, lsa[2], lsa[3]))
                if 'v_line_pallet_body' in viz_color_dict:
                    mask_edges[lsa[0], lsa[1]:lsa[2], :] = viz_color_dict['v_line_pallet_body']
            else:  # 向上查找时，只要有一行不符合，则视为断开，立即停止查找
                break
            cur_y, cur_x = _y, _x

        # 卡板整体
        pointy_pallet_body_top_line = line_seg_array[index_line_pallet_leg_top_line - (c - 1), 0]
        # print('c', c)
        # print('pointy_pallet_leg_top_line:', pointy_pallet_leg_top_line)
        # print('pointy_pallet_body_top_line:', pointy_pallet_body_top_line)
        #
        # ----- 将检测到的卡板脚和卡板上方木板的对应线段合并保存
        # line_pallet_body_array = np.asarray(line_pallet_body_list)
        # print('line_pallet_leg_array', line_pallet_leg_array.shape,
        # 	'line_pallet_body_array', line_pallet_body_array.shape)
        if len(line_pallet_body_list) > 0:
            line_pallet_array = np.concatenate((line_pallet_leg_array, np.asarray(line_pallet_body_list)), axis=0)
        else:
            line_pallet_array = line_pallet_leg_array
        # print(line_pallet_leg_array.shape, np.asarray(line_pallet_body_list).shape, line_pallet_array.shape)
        # print(index_min_col, c)
        # --------------------------------------------------------------------------------------------------------
        #
        # 以找到的所有卡板侧边点像素位置的均值，作为卡板侧边最终位置
        pointx_pallet_edge = np.round(np.mean(line_pallet_array[:, 1])).astype(np.int32)

        if 'h_line_pallet' in viz_color_dict:
            cv2.line(mask_edges, (pointx_pallet_edge, 0), (pointx_pallet_edge, depth.shape[0]),
                     viz_color_dict['h_line_pallet'], 3)  # (BGR)

        # ----- 卡板上方的货物的边界检测： 卡板上方的所有检测到线，全部视为货物的线
        line_goods_array = np.asarray(line_seg_array[:index_line_pallet_leg_top_line])
        # print(line_goods_array)

        if 'point_goods_edge' in viz_color_dict:
            for j in range(line_goods_array.shape[0]):
                _pt_color = viz_color_dict['point_goods_edge']
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1], :] = _pt_color  # MediumVioletRed
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1] - 1, :] = _pt_color  #
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1] + 1, :] = _pt_color  #
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1] - 2, :] = _pt_color  #
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1] + 2, :] = _pt_color  #
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1] - 3, :] = _pt_color  #
                mask_edges[line_goods_array[j, 0], line_goods_array[j, 1] + 3, :] = _pt_color  #

        # ----- 找到货物边界点中最左边的点，即最靠近左边的列，亦为列的索引的最小值
        # left_overstep_points = np.amin(line_goods_array[:, 1])  # x
        # print('line_goods_array.shape', line_goods_array.shape)
        if line_goods_array.shape[0] > 0:
            overstep_points_idx = np.argmin(line_goods_array[:, 1])
            # ----- 获取对应点的坐标
            overstep_points_y = line_goods_array[overstep_points_idx, 0]
            overstep_points_x = line_goods_array[overstep_points_idx, 1]
        else:  # 增加异常处理
            overstep_points_idx = -1
            overstep_points_y = 0
            overstep_points_x = 0

        # print(line_goods_array[overstep_points_idx, :])
        # print(overstep_points_y)
        # if find more than one points
        if 'h_line_overstep' in viz_color_dict:
            # mask_edges[:, left_overstep_points, :] = [0, 0, 139]  # DarkBlue
            cv2.line(mask_edges, (overstep_points_x, 0), (overstep_points_x, depth.shape[0]),
                     viz_color_dict['h_line_overstep'], 3)  # (BGR)
        if 'point_overstep' in viz_color_dict:
            _pt = (overstep_points_x, overstep_points_y)
            _r = 5
            cv2.circle(mask_edges, _pt, _r, viz_color_dict['point_overstep'], _r)  # 圆的主体
            cv2.circle(mask_edges, _pt, _r + 1, [255, 255, 255], 2)  # 圆的描边
            cv2.circle(mask_edges, _pt, 1, [0, 0, 0], 2)  # 圆的中心

        # TODO 增加每个变量的详细说明
        result_dict = {
            'state': 'OK',
            'info': 'OK',
            # 'point_overstep': (overstep_points_x, overstep_points_y),
            # 'point_overstep': (overstep_points_x, overstep_points_y),
            'pointx_goods': overstep_points_x,
            'pointy_goods': overstep_points_y,
            'line_goods_array': line_goods_array,
            'line_pallet_array': line_pallet_array,
            'line_pallet_leg_list': line_pallet_leg_list,
            'line_pallet_body_list': line_pallet_body_list,
            # 'pointx_pallet_edge': pointx_pallet_edge,
            'pointx_pallet': pointx_pallet_edge,
            'pointy_pallet': pointy_pallet_body_top_line,
            'pointy_pallet_leg_top_line': pointy_pallet_leg_top_line,
            # 'pointy_pallet_body_top_line:': pointy_pallet_body_top_line,
            'mask': mask_edges
        }
        return result_dict

    # ##################################################################
    # 算法说明：
    # 梯度 < 阈值，则认为该部分的数据为连续
    # ##################################################################
    @staticmethod
    def find_line_segment(depth, idx_h, params):
        # ---------------------------------------------------
        depth_line = depth[idx_h, :]
        line_segment_list = list()
        #
        # ------- 一阶导数
        g1 = np.gradient(depth_line)
        # ------- 获取满足条件（梯度 < 阈值）的点，记录其位置
        idx_g1 = np.where(abs(g1) < params['T_GRADIENT'])
        # 数据需为有效数据
        idx_valid = np.where(np.logical_and(depth_line != 0, depth_line < params['T_DIST_MAX']))
        # print('len(idx_valid)', idx_valid[0].size, 'len(idx_g1)', idx_g1[0].size)
        index_line = np.intersect1d(idx_g1, idx_valid)  # numpy array

        # ------- 对该行数据进行初步过滤，跳过总的有效数据点 < T_NUMBER_OF_VALID_POINTS_PER_LINE的行
        if index_line.size > params['T_NUMBER_OF_VALID_POINTS_PER_LINE']:
            # ------- 划分线段，获取每个线段的起点，终点，长度，
            # ------- 同时，过滤线段：剔除长度小于阈值T_LINE_LENGTH的线段
            # line_segment_list is list of tuple, each element of the list stores (start, end, length)
            # 'start' and 'end' are the ends of the line segment
            # and they are the index of the corresponding points in the line.
            # index相邻，则梯度为1，故取>1的值即可
            line_segment_list = Detector.cut_line_segment(idx_h, index_line, params, 1.1)
            # print('line_segment_list info:', len(line_segment_list), line_segment_list)
        return line_segment_list


    # #####################################################
    # cut_line_segment:
    # input: 'index_line', index of each row
    #        't_line_min_length', the min length of the line belongs to pallet
    #        't_line_gradient', 用于判断两条线是否为相邻（垂直方向上连续）
    # output: list of tuple(start, end, length)
    @staticmethod
    def cut_line_segment(idx_h, index_line, params, t_line_gradient):
        t_line_min_length = params['T_LINE_MIN_LENGTH']
        line_seg_list = list()
        # print('len(index_line)', len(index_line))
        # 再次index的梯度，判断是否连续
        g_index = np.gradient(index_line)
        idx = 0
        while idx < len(index_line):
            start = index_line[idx]
            while abs(g_index[idx]) < t_line_gradient:
                if idx < len(index_line) - 2:
                    idx += 1
                else:
                    break
            end = index_line[idx]
            idx += 1
            if t_line_min_length < end - start + 1:
                # line_seg_list.append((start, end, end-start+1))  # tuple(start, end, length)
                line_seg_list.append((idx_h, start, end, end - start + 1))  # tuple(idx_h, start, end, length)
                # print('Start to end: {}---{}, length:{}'.format(start, end, end-start+1))
        return line_seg_list

    @staticmethod
    def distance_of_point_to_plane(plane_params, points):
        x, y, z = points
        plane_z = Detector.plane(x, y, plane_params)
        diff = abs(plane_z - z)
        return diff

    @staticmethod
    def plane(x, y, plane_params):
        a = plane_params[0]
        b = plane_params[1]
        c = plane_params[2]
        z = a * x + b * y + c
        return z

    @staticmethod
    def error(plane_params, points):  # distance_of_point_to_plane
        result = 0
        for (x, y, z) in points:
            plane_z = Detector.plane(x, y, plane_params)
            diff = abs(plane_z - z)
            result += diff ** 2
        return result

    @staticmethod
    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    @staticmethod
    def viz_line_segment(image, line_mask):
        # --- 显示线段聚类的结果
        image = cv2.addWeighted(image, 1, line_mask, 0.3, 0)
        return image

    @staticmethod
    def viz_edge_points(image, pt_x, color):
        # --- 显示卡板/货物的边缘
        image = cv2.line(image, (pt_x, 0), (pt_x, image.shape[0]), color, 2)  # (BGR)
        return image

    @staticmethod
    def viz_overstep_point(image, pt, color, r=5):
        # 突出的那一个点
        # pt = results['point_overstep']
        cv2.circle(image, pt, r, color, r)  # 圆的主体
        cv2.circle(image, pt, 2 * r - 1, [0, 0, 200], 1)  # 圆的描边
        cv2.circle(image, pt, 1, [0, 0, 0], 1)  # 圆的中心
        return image

    @staticmethod
    def viz_image_assemble(image1, image2, image3):
        _h, _w, _ = image1.shape
        img_assembled = np.ones((_h, _w * 2, 3), dtype=np.uint8) * 255
        # image 1
        img_assembled[:, :_w, :] = image1
        # image 2
        ws = _w
        hs = int(image2.shape[0] * _w / image2.shape[1])
        res_figure = cv2.resize(image2, (ws, hs), interpolation=cv2.INTER_CUBIC)
        img_assembled[_h - hs:, _w:_w + ws, :] = res_figure

        wu = _w
        hu = int(image3.shape[0] * _w / image3.shape[1])
        res_imgu = cv2.resize(image3, (wu, hu), interpolation=cv2.INTER_CUBIC)
        # print(_h, hs, hu)
        img_assembled[_h - hs - hu:_h - hs, _w:_w + wu, :] = res_imgu
        return img_assembled

    # @staticmethod
    # @staticmethod
    # @staticmethod


if __name__ == '__main__':
    def get_depth_filepaths(root):
        d_filepaths = glob.glob(root + '/*depth*.png')  # List all files in data folder
        d_filepaths.sort()
        print("Get {} depth files in '{}'.".format(len(d_filepaths), root))
        return d_filepaths
    time_start = datetime.datetime.now()

    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20190909')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20190918')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\debug')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20191920\\20cm')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20191920\\30cm')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20190921')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image1\\plastic_pallet_image')
    # depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_imgae_20190923_SF\\10cm')
    depth_filepaths = get_depth_filepaths('C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20190924')

    # -----------------------------------
    # m_detector = Detector('PLASTIC')
    m_detector = Detector('PLASTIC')

    # exit(0)

    for depth_filepath in depth_filepaths:
        depth_dir = os.path.dirname(depth_filepath)
        depth = os.path.basename(depth_filepath)
        rgb_base = depth.replace('depth', 'rgb')
        rgb_filepath = os.path.join(depth_dir, rgb_base)
        # m_detector.logger.info("Depth file:{}, RGB file:{}".format(depth_filepath, rgb_filepath))
        print('depth_filepath: ', depth_filepath)

        depth_img = m_detector.read_depth(depth_filepath)
        rgb_img = m_detector.read_rgb(rgb_filepath)

        # ---
        depth_img = np.rot90(depth_img, axes=(1, 0))
        rgb_img = np.rot90(rgb_img, axes=(1, 0))

        # flag_flip = True  # 右侧叉齿的图像需要将图像左右翻转
        flag_flip = False  # 右侧叉齿的图像需要将图像左右翻转
        # ---- 图像翻转：该算法只能处理左侧插齿位置对应的图像，若要处理右侧的图像，将图像左右翻转即可
        if flag_flip:
            depth_img = cv2.flip(depth_img, 1, dst=None)  # 水平镜像
            rgb_img = cv2.flip(rgb_img, 1, dst=None)  # 水平镜像

        # --------------------------------------------------------------------------
        #
        #
        # colorpc = Detector.get_point_cloud(params=m_detector.oversize_detection_params, depth=depth_img, rgb=rgb_img)
        # Detector.save_point_cloud(colorpc, depth_filepath[:-4] + '_cpc.ply')

        # --------------------------------------------------------------------------
        # ------ save the pallet leg detection result to ply data.
        # dist_x_pixel, dist_x_mm = m_detector.run(depth_img, plyname=depth_filepath[:-4] + 'pallet_floor.ply')
        #
        # ------ save the detection result to RGB image.
        # dist_x_pixel, dist_x_mm = m_detector.run(depth_img, rgb=rgb_img, vizname=depth_filepath[:-4]+'viz.jpg')
        #
        # ------ only output the numbers
        # dist_x_pixel, dist_x_mm = m_detector.run(depth_img)

        dist_x_pixel, dist_x_mm, result_details = m_detector.run(depth_img)

        m_detector.viz_result(depth_img, rgb_img, result_details, vizname=depth_filepath[:-4]+'viz.jpg')

        #
        print("Over size detection:{} pixels, {} mm".format(dist_x_pixel, dist_x_mm))

    print('Processed finished in: {}'.format(datetime.datetime.now() - time_start))



