#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LI.Jie@Kiktech
# Feb. 20, 2020


# import math
import numpy as np

from TOF_CONFIG import TOF_PARAMS_OBSTACLE_DETECTION as P_OD

from kik_pcl import mark_certain_region_in_voxel, mark_obstacle_in_voxel, mark_detected_obstacle_in_voxel

from kik_pcl import depth2voxel, get_color_point_cloud, save_point_cloud

# from kik_pcl import point_cloud2voxel

from kik_pcl import get_point_cloud

# from c_kik_pcl import get_point_cloud

from c_kik_pcl import point_cloud2voxel


import imageio
import datetime
from depth_encoder_decoder import decode_depth


class ObstacleDetection(object):
    def __init__(self, cell_length=20, grid_length=20, min_obstacle_height=50):
        """
        :param cell_length: 计算过程中，每个体素的尺寸，单位mm，越小精度越高，速度越慢
        :param grid_length: 用户定义的地面栅格地图的栅格尺寸，单位mm。grid_length必须是cell_length的整数倍
        :param min_obstacle_height: 用于确定三维栅格中，障碍物的最小高度的阈值，单位为mm。高于该阈值的体素，视为障碍，否则视为地面
        """

        # ######################### TOF 相机参数
        self.CAM_K = P_OD['CAM_K']
        # ---- 视野盲区，剔除盲区内的噪音数据
        self.TOF_Min_Detection_Range = P_OD['TOF_Min_Detection_Range']  # 最小检测距离，单位mm
        # self.TOF_Max_Detection_Range = 5000  # 最大检测距离
        
        # ######################### 用于校准检测算法的参数
        # TOF中心距离地面的高度，单位mm, 并进行垂直方向调整，使得地面位于XOZ平面
        self.TOF_Height = P_OD['TOF_Height']  # 通过校准提供旋转矩阵R与TOF中心距离地面的高度
        self.R = P_OD['R']

        # ######################### 依据需求而设置的参数
        # 避障检测范围的宽度（左右，对应X轴），长度（前后，对应Z轴），高度（上下，对应Y轴），单位mm
        self.TOF_Detection_Area_width = P_OD['TOF_Detection_Area_width']  # X mm,必须是50的整数倍
        self.TOF_Detection_Area_Length = P_OD['TOF_Detection_Area_Length']  # Z mm
        self.TOF_Detection_Area_Height = P_OD['TOF_Detection_Area_Height']  # Y mm
        # -----

        # ######################### 算法内部参数
        # 设置每个栅格的边长，单位mm
        self.TOF_Grid_Unit_Length = cell_length  # mm
        # 超参数，用于确定三维栅格中，地面体素的最小高度的阈值，单位为mm。高于该阈值的体素，视为障碍，否则视为地面
        self.Ground_Threshold = min_obstacle_height  # mm
        # 超参数，用于确定三维栅格中，地面体素的最小高度的阈值，单位为体素个数。高于该阈值的体素，视为障碍，否则视为地面
        self.Ground_Threshold_Grids = self.Ground_Threshold // self.TOF_Grid_Unit_Length  # grid

        # 有效体素内的点云的最小数目
        self.Min_Point_Number = 1  # point

        # 检测区域对应的三维栅格的分辨率
        self.grid_size = [self.TOF_Detection_Area_width // self.TOF_Grid_Unit_Length,
                          self.TOF_Detection_Area_Height // self.TOF_Grid_Unit_Length,
                          self.TOF_Detection_Area_Length // self.TOF_Grid_Unit_Length]

        # 设置offset: x 需要检测的宽度的一半，单位mm
        # 设置offset: y 为TOF中心距离地面的高度，单位mm
        self.offset_xyz = [self.TOF_Detection_Area_width / 2, self.TOF_Height, 0]  # [x, y, z] mm

        # ######################### 计算其他参数
        # ---- 用于剔除视野盲区内的噪音数据
        self.min_z_grid = self.TOF_Min_Detection_Range // self.TOF_Grid_Unit_Length

        # ----- 输出栅格的采样率
        # grid_length: 用户定义的地面栅格地图的栅格尺寸，单位mm
        # self.sample_rate: 将内部结果进行采样，获得用户定义的输出分辨率
        assert grid_length % self.TOF_Grid_Unit_Length == 0
        self.sample_rate = grid_length // self.TOF_Grid_Unit_Length

    def run(self, depth_uint16, filename=None):
        flag_viz_obstacle = True
        # flag_viz_obstacle = False

        grid_size = self.grid_size
        unit = self.TOF_Grid_Unit_Length
        o_xyz = self.offset_xyz

        #
        # voxel_count, index_depth2grid, index_grid2depth = depth2voxel(depth_uint16, unit=unit, gridsize=grid_size,
        #                                                               offset=o_xyz, cam_k=cam_k)

        # ----- 转换为点云
        point_cloud = get_point_cloud(depth_uint16, self.CAM_K)  # 0.012s

        # ----- 点云旋转
        rotated_point_cloud = np.dot(np.reshape(point_cloud, (-1, 3)), self.R).reshape(point_cloud.shape)  # 0.01s

        # ----- 将点云体素化
        #
        voxel_count, index_depth2grid, index_grid2depth = point_cloud2voxel(rotated_point_cloud, unit, grid_size, o_xyz)
        #print(index_depth2grid.shape)
        # # ----- visualise: 显示所有体素，并将障碍物标记为不同颜色
        if filename:
            voxel_ply = mark_obstacle_in_voxel(voxel_count, self.Ground_Threshold_Grids, self.Min_Point_Number)
            save_point_cloud(voxel_ply, filename)

        # ----- 剔除视野盲区内的噪音数据
        voxel_count[:, :, :self.min_z_grid] = 0

        # ----- 在栅格地图中标记有障碍的格子
        obstacle = voxel_count[:, self.Ground_Threshold_Grids + 1:, :]

        # 3 --------- 将找到的卡板区域对应的深度图的部分取出来,并标记为1，其余部分的深度图中的像素值标记为0
        if flag_viz_obstacle:
            
            obstacle_mask_2D = np.zeros(depth_uint16.shape, dtype=np.uint8)
            #print(obstacle_mask_2D.shape)
            # ---- 将超出检测区域的像素标记为255
            # ---- 沿X轴，左右两侧
            idx_x1 = index_depth2grid[:, :, 0] <= 1
            
            idx_x2 = index_depth2grid[:, :, 0] >= grid_size[0] - 2
            obstacle_mask_2D[idx_x1] = 255
            obstacle_mask_2D[idx_x2] = 255
            # ---- 沿Z轴
            # idx_z1 = index_depth2grid[:, :, 2] <= 1
            # idx_z2 = index_depth2grid[:, :, 2] >= grid_size[2] - 2
            # obstacle_mask_2D[idx_z1] = 255
            # obstacle_mask_2D[idx_z2] = 255
        else:
            obstacle_mask_2D = None

        # ----- 构建地面栅格地图, 0.04s
        # high_resolution_obstacle_2darray = np.ones(grid_size[0])
        
        high_resolution_obstacle_2darray = np.zeros(grid_size[0], dtype=np.int32)
        for _idx in range(grid_size[0]):
            # 逐条对垂直扫描线进行判断
            
            closest_obstacle = obstacle[_idx, :, :]
            #print(closest_obstacle.shape)
            idx_arr = np.where(closest_obstacle >= self.Min_Point_Number)
            
            # print(idx_arr[1].size)
            if idx_arr[1].size > 0:
                high_resolution_obstacle_2darray[_idx] = np.min(idx_arr[1])  # find the closest obstacle along Z
                # high_resolution_obstacle_2darray[_idx] = idx_arr[1][0]  # find the closest obstacle along Z
                # print(high_resolution_obstacle_2darray[_idx])

                if flag_viz_obstacle:  # 障碍物较多时，该过程耗时
                    # ---- 获取障碍物所在的体素
                    tx = _idx
                    _i = np.argmin(idx_arr[1])
                    ty = idx_arr[0][_i]
                    tz = idx_arr[1][_i]

                    # ---- 获取障碍物所占的体素区域在对应的深度图中的位置索引
                    px_min = tx - 1
                    py_min = ty
                    pz_min = tz
                    px_max = tx + 1
                    py_max = ty + 1
                    pz_max = tz + 1
                    # 位于该栅格内的所有像素
                    idx_1 = (px_min <= index_depth2grid[:, :, 0]) & (index_depth2grid[:, :, 0] <= px_max)
                    # idx_2 = (py_min <= index_depth2grid[:, :, 1]) & (index_depth2grid[:, :, 1] <= py_max)
                    idx_2 = py_min <= index_depth2grid[:, :, 1]
                    # idx_3 = (pz_min <= index_depth2grid[:, :, 2]) & (index_depth2grid[:, :, 2] <= pz_max)
                    idx_3 = pz_min <= index_depth2grid[:, :, 2]

                    obstacle_mask_2D[idx_1 & idx_2 & idx_3] = 1  # 将障碍物对应像素标记为1

            # if idx_arr[1].size == 0:
            #     high_resolution_obstacle_2darray[_idx] = 0  # mark: no obstacle in this slice
            # print(type(idx_arr), idx_arr[1])

        # ----- 按照用户定义，将检测结果按照定义的栅格地图的大小，进行重采样
        # sr = 5
        sr = self.sample_rate
        user_defined_resolution_obstacle_2darray = high_resolution_obstacle_2darray / sr
        obstacle_grid_slice = np.split(user_defined_resolution_obstacle_2darray, grid_size[0] // sr)

        index_of_obstacle_in_each_gird_slice = np.zeros(grid_size[0] // sr, dtype=int)
        for _idx in range(grid_size[0] // sr):
            # print(obstacle_grid_slice[_idx])
            # 查找障碍物所在的位置
            idx_arr = np.where(obstacle_grid_slice[_idx] > 0)
            # print(type(idx_arr), idx_arr)
            # print(idx_arr[0].size)
            if idx_arr[0].size > 0:  # 查找除0以外的最小值
                index_of_obstacle_in_each_gird_slice[_idx] = obstacle_grid_slice[_idx][idx_arr[0][0]]

        # 可视化检验：先将检测区域标记为绿色，再将有障碍的地方标记为红色，将检测的结果与原始点云叠加。
        # if filename:
        #     # voxel_ply = mark_obstacle_in_voxel(voxel_count, self.Ground_Threshold_Grids, self.Min_Point_Number)
        #     voxel_ply = mark_detected_obstacle_in_voxel(voxel_count,
        #                                                 self.Ground_Threshold_Grids,
        #                                                 index_of_obstacle_in_each_gird_slice,
        #                                                 self.sample_rate,
        #                                                 self.Min_Point_Number)
        #     save_point_cloud(voxel_ply, filename)
        return index_of_obstacle_in_each_gird_slice, obstacle_mask_2D


if __name__ == '__main__':
    fileroot = 'C:/Users/LiJie/Documents/kik/kik_tof_UnloadingSpace/'
    # RGBD_filename = fileroot+'tof_data_20200217/50_1/2020-02-17_17.40.02.png'
    # RGBD_filename = fileroot+'tof_data_20200217/50_1/2020-02-17_17.40.17.png'
    # RGBD_filename = fileroot+'tof_data_20200217/70_1/2020-02-17_17.22.09.png'
    # ---- 避障，直行
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_one_1/2020-02-19_16.57.38.png'
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_one_1/2020-02-19_16.57.51.png'
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_one_1/2020-02-19_16.57.52.png'
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_one_1/2020-02-19_16.57.53.png'
    # ---- 避障，拐弯
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_three/2020-02-19_17.40.41.png'
    # obj = ((496, 290), (582, 387))
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_three/2020-02-19_17.40.42.png'
    # obj = ((406, 290), (516, 417))
    # RGBD_filename = fileroot+'tof_data_20200219_2/scenario_three/2020-02-19_17.40.43.png'
    # obj = ((133, 317), (295, 474))

    # ---- 库位，场景1，第一排空泳道
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_one_50_1/2020-02-19_16.01.14.png'
    # ---- 库位，场景2，第一排泳道，间隔堆放货物
    RGBD_filename = fileroot+'tof_data_20200219_1/scenario_two_50_1/2020-02-19_16.27.04.png'
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_two_50_1/2020-02-19_16.27.05.png'
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_two_50_1/2020-02-19_16.27.06.png'
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_two_50_1/2020-02-19_16.27.07.png'
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_two_50_1/2020-02-19_16.27.08.png'
    # ---- 库位，场景3，第一排泳道堆满货物
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_three_50_1/2020-02-19_16.34.27.png'

    # -------------------------
    # 障碍高度 3cm
    # RGBD_filename = fileroot+'tof_data_20200221/3cm/2020-02-21_15.35.31.png'
    # RGBD_filename = fileroot+'tof_data_20200221/3cm/2020-02-21_15.35.32.png'
    # RGBD_filename = fileroot+'tof_data_20200221/3cm/2020-02-21_15.35.33.png'
    # 4cm
    # RGBD_filename = fileroot+'tof_data_20200221/4cm/2020-02-21_15.32.41.png'
    # 5cm
    # RGBD_filename = fileroot+'tof_data_20200221/5cm/2020-02-21_15.30.00.png'
    # RGBD_filename = fileroot+'tof_data_20200221/5cm/2020-02-21_15.30.01.png'
    # 6cm
    # RGBD_filename = fileroot+'tof_data_20200221/6cm/2020-02-21_15.27.14.png'
    # RGBD_filename = fileroot+'tof_data_20200221/6cm/2020-02-21_15.27.15.png'
    # 8cm
    # RGBD_filename = fileroot+'tof_data_20200221/8cm/2020-02-21_15.24.16.png'
    # RGBD_filename = fileroot+'tof_data_20200221/8cm/2020-02-21_15.24.17.png'
    # RGBD_filename = fileroot+'tof_data_20200221/8cm/2020-02-21_15.24.18.png'
    # RGBD_filename = fileroot+'tof_data_20200221/8cm/2020-02-21_15.24.19.png'
    # 10cm
    # RGBD_filename = fileroot+'tof_data_20200221/10cm/2020-02-21_15.43.12.png'
    # 15cm
    # RGBD_filename = fileroot+'tof_data_20200221/15cm/2020-02-21_15.40.44.png'
    # 20cm
    # RGBD_filename = fileroot+'tof_data_20200221/20cm/2020-02-21_15.38.40.png'

    # -------------------
    # RGBD_filename = fileroot+'tof_20200302/tof_data/5cm/2020-03-02_16.16.06.png'
    # RGBD_filename = fileroot+'tof_20200302/tof_data/8cm/2020-03-02_16.11.50.png'
    # RGBD_filename = fileroot+'tof_20200302/tof_data/8cm/2020-03-02_16.11.51.png'
    # RGBD_filename = fileroot+'tof_20200302/tof_data/8cm/2020-03-02_16.11.52.png'


    RGBD_img = imageio.imread(RGBD_filename)
    depth_3uint8 = RGBD_img[:, 0:RGBD_img.shape[1] // 2, :]  # 480x1280-->480x640
    rgb_img = RGBD_img[:, RGBD_img.shape[1] // 2:, :]  # 480x1280-->480x640
    input_depth_uint16 = decode_depth(depth_3uint8, order='RGB')

    # cply = get_point_cloud(input_depth_uint16, PARAMS['CAM_K'])
    # save_point_cloud(cply, RGBD_filename[:-4] + '_point.ply')

    # cply = get_color_point_cloud(rgb_img, input_depth_uint16, PARAMS['CAM_K'])
    # save_point_cloud(cply, RGBD_filename[:-4] + '_colorpoint.ply')

    # TOF_Detection_Area_width/cell_length 必须是 TOF_Detection_Area_width/grid_length的整数倍
    cell_length = 20  # 建议设置为20或更大的值
    grid_length = 100  # 需设置为cell_length的整数倍
    min_obstacle_height = 50
    m_tof_obstacle_detection = ObstacleDetection(cell_length, grid_length, min_obstacle_height)

    time_start = datetime.datetime.now()
    filename = None
    # filename = RGBD_filename
    # filename = RGBD_filename[:-4] + '_voxel{}mm.ply'.format(min_obstacle_height)
    # filename = RGBD_filename[:-4] + '_detection_{}_{}_{}.ply'.format(cell_length, grid_length, min_obstacle_height)
    obstacle_gz, obstacle_mask_2D = m_tof_obstacle_detection.run(input_depth_uint16, filename)

    # 标记超出检测范围的区域
    idx_region_edge_2D = np.array(np.where(obstacle_mask_2D == 255))
    if idx_region_edge_2D.size > 0:
        rgb_img[idx_region_edge_2D[0], idx_region_edge_2D[1], :] = (0, 0, 255)
        depth_3uint8[idx_region_edge_2D[0], idx_region_edge_2D[1], :] = (0, 0, 255)
    # ---- 可视化：将障碍物检测结果的mask绘制到图像中
    idx_obstacle_2D = np.array(np.where(obstacle_mask_2D == 1))
    if idx_obstacle_2D.size > 0:
        #
        rgb_img[idx_obstacle_2D[0], idx_obstacle_2D[1], :] = (255, 0, 0)
        depth_3uint8[idx_obstacle_2D[0], idx_obstacle_2D[1], :] = (255, 0, 0)
        imageio.imwrite(RGBD_filename[:-4] + '_rgb.jpg', rgb_img)
        imageio.imwrite(RGBD_filename[:-4] + '_d.jpg', depth_3uint8)
        #
        color_mask = np.zeros(rgb_img.shape, dtype=np.uint8)
        import cv2
        masked_rgb = cv2.addWeighted(rgb_img, 1, color_mask, 0.5, 0)
        imageio.imwrite(RGBD_filename[:-4] + '_masked.jpg', masked_rgb)

        print('saved visualization images')

    print(obstacle_gz)

    print('Processed finished in: {}'.format(datetime.datetime.now() - time_start))
