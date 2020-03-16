#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LI.Jie@Kiktech
# Oct 6, 2019


import logging
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


# ----------------------------------------------------------------------------------------------
import os
import time
import imageio
import numpy as np
from depth_encoder import encode_depth

# ----- record the Depth and RGB images


class RecordTof(object):
    def __init__(self, root):
        self.timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())  # timestamp for the folder
        print("Timestamp: ", self.timestamp)

        if not os.path.exists(root):  # 判断当前路径是否存在，没有则创建new文件夹
            os.makedirs(root)
        self.path = os.path.join(root, self.timestamp)
        os.makedirs(self.path)

    def save_data_to_disk(self, depth_16bit, RGB):
        # print(depth_16bit.shape, RGB.shape)
        depth_8bit_3channel = encode_depth(depth_16bit)
        img_out = np.concatenate((depth_8bit_3channel, RGB), axis=1)  # (H, W1+W2, C)

        # print(img_out.shape)
        file_timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())  # timestamp for the file

        imageio.imwrite(os.path.join(self.path, file_timestamp+'.png'), img_out)



