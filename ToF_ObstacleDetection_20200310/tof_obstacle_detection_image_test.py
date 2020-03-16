#!/usr/bin/env python
# -*- coding: utf-8 -*-

import imageio
import numpy as np
import cv2
import glob
import datetime

from tof_obstacle_detection.tof_ObstacleDetection import ObstacleDetection
from tof_obstacle_detection.depth_encoder_decoder import decode_depth

def get_depth_filepaths(root):
    d_filepaths = glob.glob(root + '/*.png')  # List all files in data folder
    d_filepaths.sort()
    print("Get {} depth files in '{}'.".format(len(d_filepaths), root))
    return d_filepaths

if __name__ == '__main__':
    fileroot = get_depth_filepaths('tof_obstacle_test_images')
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
    # RGBD_filename = fileroot+'tof_data_20200219_1/scenario_two_50_1/2020-02-19_16.27.04.png'
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
    # cply = get_point_cloud(input_depth_uint16, PARAMS['CAM_K'])
    # save_point_cloud(cply, RGBD_filename[:-4] + '_point.ply')

    # cply = get_color_point_cloud(rgb_img, input_depth_uint16, PARAMS['CAM_K'])
    # save_point_cloud(cply, RGBD_filename[:-4] + '_colorpoint.ply')

    # TOF_Detection_Area_width/cell_length 必须是 TOF_Detection_Area_width/grid_length的整数倍
    cell_length = 20  # 建议设置为20或更大的值
    grid_length = 100  # 需设置为cell_length的整数倍
    min_obstacle_height = 50
    m_tof_obstacle_detection = ObstacleDetection(cell_length, grid_length, min_obstacle_height)

    for filepath in fileroot:
        print(filepath)
        RGBD_img = imageio.imread(filepath)
        depth_3uint8 = RGBD_img[:, 0:RGBD_img.shape[1] // 2, :]  # 480x1280-->480x640
        rgb_img = RGBD_img[:, RGBD_img.shape[1] // 2:, :]  # 480x1280-->480x640
        input_depth_uint16 = decode_depth(depth_3uint8, order='RGB')


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
            # imageio.imwrite(RGBD_filename[:-4] + '_rgb.jpg', rgb_img)
            # imageio.imwrite(RGBD_filename[:-4] + '_d.jpg', depth_3uint8)
            #
            color_mask = np.zeros(rgb_img.shape, dtype=np.uint8)
            color_mask[idx_obstacle_2D[0],idx_obstacle_2D[1],:] = (255,0,0)
            import cv2
            masked_rgb = cv2.addWeighted(rgb_img, 1, color_mask, 0.5, 0)
            # imageio.imwrite(RGBD_filename[:-4] + '_masked.jpg', masked_rgb)

            print('saved visualization images')

        print(obstacle_gz)

        print('Processed finished in: {}'.format(datetime.datetime.now() - time_start))
        cv2.imshow('demo', masked_rgb)
        cv2.waitKey()
