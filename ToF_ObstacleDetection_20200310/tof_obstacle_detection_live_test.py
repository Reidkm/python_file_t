#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from axonopenni import openni2
import numpy as np
import cv2, time, ctypes
from axonopenni import _openni2 as c_api
from tof_oversize_pallet_detector_3 import Detector
from utils_1 import RecordTof
import imageio
import glob
import datetime

from tof_obstacle_detection.tof_ObstacleDetection import ObstacleDetection
from tof_obstacle_detection.depth_encoder_decoder import decode_depth


SHOW_RESULT = True

class AXonDemo(object):

    def __init__(self):
        openni2.initialize()

    def open(self):
        self.dev = openni2.Device.open_any()
        dev_info = self.dev.get_device_info()
        depth_info = self.dev.get_sensor_info(openni2.SENSOR_DEPTH)
        print("设备信息：", dev_info)
        print("Depth传感器类型：%r, 模式：%r" % (depth_info.sensorType, depth_info.videoModes))

    def select_ImageRegistration(self, mode=0):

        self.dev.is_image_registration_mode_supported(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        result = self.dev.get_image_registration_mode()
        print("当前图像配准模式：", result)

        if mode == 0:
            obj = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_OFF
        elif mode == 1:
            obj = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR
        elif mode == 2:
            obj = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_IR_TO_COLOR
        elif mode == 3:
            obj = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_COLOR_TO_DEPTH
        elif mode == 4:
            obj = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_COLOR_UNDISTORTION_ONLY
        else:
            print("Test ImageRegistration Mode Error！")
            return False

        res = self.dev.is_image_registration_mode_supported(mode)
        print("Support Result:", res)
        if res:
            try:
                c_api.oniDeviceSetProperty(self.dev._handle, c_api.ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION,
                                           ctypes.byref(obj), ctypes.sizeof(obj))
                result = self.dev.get_image_registration_mode()
            except Exception as e:
                print("*Test Registration Error!: %r " % e)
                return False
            print("设置后的图像配准模式：", result)
            if obj == result:
                return True
            else:
                print("图像配准设置失败")
                return False
        else:
            print("Is_image_registration_mode_supported Error")
            return False

    def set_dcsync(self):
        try:
            result = self.dev.get_depth_color_sync_enabled()
            print("当前Depth to Color帧同步使能为：", result)

            enable = self.dev.set_depth_color_sync_enabled(True)
            result2 = self.dev.get_depth_color_sync_enabled()
            print("设置帧同步后使能为：", result2)

        except Exception as e:
            print("*TEST test_dcsync Error!：%r" % e)
            return False
        else:
            if enable.value == 0:
                return True
            else:
                return False

    def get_depth_value_unit(self):
        format = self.dev.get_sensor_info(openni2.SENSOR_DEPTH).videoModes[0].pixelFormat
        unit = openni2.get_depth_value_unit(format)
        print("\n*当前深度数据的单位为%.2fmm\n" % unit)
        return unit

    def create_streams(self):
        dvs = self.dev.create_depth_stream()
        cvs = self.dev.create_color_stream()
        rvs = self.dev.create_ir_stream()
        streams = [dvs, cvs, rvs]
        return streams

    def close_streams(self, streams):
        for vs in streams:
            vs.close()

    def start_streams(self, streams, sec=5):
        for vs in streams:
            vs.start()

        start_time = time.time()
        m_detector = Detector('PLASTIC')
        m_tof_record = RecordTof('./tof_data')
        index = 10 
	cell_length = 20  # 建议设置为20或更大的值
    	grid_length = 100  # 需设置为cell_length的整数倍
    	min_obstacle_height = 50
    	m_tof_obstacle_detection = ObstacleDetection(cell_length, grid_length, min_obstacle_height)
        while True:
            #if (time.time() - start_time) >= sec:
            #    print("拉流总耗时：%.2f秒" % (time.time() - start_time))
            #    cv2.destroyAllWindows()
            #    break

            openni2.wait_for_any_stream(streams)

            dframe = streams[0].read_frame()
            cframe = streams[1].read_frame()
            rframe = streams[2].read_frame()
            cframe_data = None
            dpt = None
            if dframe!=None:
                dframe_data = np.array(dframe.get_buffer_as_triplet()).reshape([480, 640, 2])
                dframe_data_1 = np.array(dframe.get_buffer_as_uint16()).reshape([480, 640])
                dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')   #float32 
                dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')   #float32 
                dpt2 *= 255
                #dpt = dpt1 + dpt2
		depth_3uint8 = dpt1 + dpt2
                # cv2.imshow('dpt', dpt)
                #cv2.moveWindow('dpt', 650, 10)
                # cv2.waitKey(1)

            if cframe!=None:
                cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
                R = cframe_data[:, :, 0]
                G = cframe_data[:, :, 1]
                B = cframe_data[:, :, 2]
                #cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
		rgb_img = np.transpose(np.array([B, G, R]), [1, 2, 0])
                #cv2.imshow('color', cframe_data)
                #cv2.moveWindow('color', 10, 10)
                # cv2.waitKey(1)

            if rframe!=None:
                rframe_data = np.array(rframe.get_buffer_as_uint8()).reshape([480, 640])
                #cv2.imshow("IR", rframe_data)
                #cv2.moveWindow('IR', 10, 500)
                # cv2.waitKey(1)

            if SHOW_RESULT:
                time_start = datetime.datetime.now()
                filename = None
                #obstacle_gz, obstacle_mask_2D = m_tof_obstacle_detection.run(input_depth_uint16, filename)
		obstacle_gz, obstacle_mask_2D = m_tof_obstacle_detection.run(depth_3uint8, filename)

                # 
                # 标记超出检测范围的区域
                idx_region_edge_2D = np.array(np.where(obstacle_mask_2D == 255))
                if idx_region_edge_2D.size > 0:
                    rgb_img[idx_region_edge_2D[0], idx_region_edge_2D[1], :] = (0, 0, 255)
                    #depth_3uint8[idx_region_edge_2D[0], idx_region_edge_2D[1], :] = (0, 0, 255)
                # ---- 可视化：将障碍物检测结果的mask绘制到图像中
                idx_obstacle_2D = np.array(np.where(obstacle_mask_2D == 1))
                if idx_obstacle_2D.size > 0:
                    #
                    rgb_img[idx_obstacle_2D[0], idx_obstacle_2D[1], :] = (255, 0, 0)
                    #depth_3uint8[idx_obstacle_2D[0], idx_obstacle_2D[1], :] = (255, 0, 0)
                    # imageio.imwrite(RGBD_filename[:-4] + '_rgb.jpg', rgb_img)
                    # imageio.imwrite(RGBD_filename[:-4] + '_d.jpg', depth_3uint8)
                    #
                    color_mask = np.zeros(rgb_img.shape, dtype=np.uint8)
                    color_mask[idx_obstacle_2D[0],idx_obstacle_2D[1],:] = (255,0,0)
                    #import cv2
                    masked_rgb = cv2.addWeighted(rgb_img, 1, color_mask, 0.5, 0)
                    # imageio.imwrite(RGBD_filename[:-4] + '_masked.jpg', masked_rgb)

                    print('saved visualization images')

                print(obstacle_gz)

                print('Processed finished in: {}'.format(datetime.datetime.now() - time_start))
                cv2.imshow('demo', masked_rgb)
            else:
                cv2.imshow('rgb', cframe_data)
                cv2.imshow('dpt', dpt)

            
            if cv2.waitKey(1) == ord('w'):
                cv2.imwrite(str(index)+'_rgb.png', cframe_data)
                cv2.imwrite(str(index)+'_dpt.png', dframe_data_1)
                print 'image saved'
                #cv2.imwrite(str(index)+".jpg", frame)
                index += 1

		
        for vs in streams:
            vs.stop()


    def close(self):
        openni2.unload()


if __name__ == "__main__":
    ax = AXonDemo()
    ax.open()
    ax.get_depth_value_unit()
    streams = ax.create_streams()
    ax.set_dcsync()   # 设置同步使能
    # ax.start_streams(streams, 3)  # 拉流，第二参数为拉流时长，单位秒
    ax.select_ImageRegistration(1)  # 设置图像对齐模式，第二个参数为对齐模式选项
    ax.start_streams(streams, 5)
    ax.close_streams(streams)
    ax.close()
