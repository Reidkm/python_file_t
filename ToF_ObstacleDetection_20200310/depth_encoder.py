#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LI.Jie@Kiktech
# Oct 6, 2019

import numpy as np


# encode depth from single channel with dtype==uint16 to 2 channels with dtype==uint8 and additional empty channel
def encode_depth(depth):
    h8 = depth >> 8
    l8 = depth - (h8 << 8)
    h8 = np.expand_dims(h8, axis=2)  # shape: (h, w) -->(h, w, 1)
    l8 = np.expand_dims(l8, axis=2)
    h8 = h8.astype(np.uint8)
    l8 = l8.astype(np.uint8)
    c = np.zeros(depth.shape+(1,), dtype=np.uint8)
    depth_hlc = np.concatenate((h8, l8, c), axis=2)  # (h, w, 3)
    # print('bin uint16', format(depth[300, 300], 'b'))
    # print('bin high 8', format(h8[300, 300, 0], 'b'))
    # print('bin low  8', format(l8[300, 300, 0], 'b'))
    return depth_hlc


def decode_depth(depth):  # hlc -- uint16
    h8 = depth[:, :, 0].astype(np.uint16)
    l8 = depth[:, :, 1].astype(np.uint16)
    # c = depth[:, :, 2].astype(np.uint16)  # empty
    depth_uint16 = (h8 << 8) + l8
    # print('bin high 8', format(h8[300, 300], 'b'))
    # print('bin low  8', format(l8[300, 300], 'b'))
    # print('bin uint16', format(depth_uint16[300, 300], 'b'))
    return depth_uint16


def decode_depth_BGR(depth):  # hlc -- uint16
    # B G R --> R(h8) G(l8) B(c) 
    h8 = depth[:, :, 2].astype(np.uint16)
    l8 = depth[:, :, 1].astype(np.uint16)
    # c = depth[:, :, 2].astype(np.uint16)  # empty
    depth_uint16 = (h8 << 8) + l8
    # print('bin high 8', format(h8[300, 300], 'b'))
    # print('bin low  8', format(l8[300, 300], 'b'))
    # print('bin uint16', format(depth_uint16[300, 300], 'b'))
    return depth_uint16


if __name__ == '__main__':
    import imageio
    import cv2

    depth_filename = 'C:\\Users\\LiJie\\Documents\\kik_tof\\pallet_RGBD\\pallet_image_20190830\\img_depth_70_L15.png'
    # depth_img = imageio.imread(depth_filename)  # <class 'numpy.uint16'>
    depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    depth_3uint8 = encode_depth(depth_img)

    # print('0 ', np.max(depth_3uint8[:, :, 0]))
    # print('1 ', np.max(depth_3uint8[:, :, 1]))
    # print('2 ', np.max(depth_3uint8[:, :, 2]))
    # imageio.imwrite('name_mask.jpg', depth_3uint8)
    cv2.imwrite('name_mask_cv2.jpg', depth_3uint8)
    depth_uint16 = decode_depth(depth_3uint8)
    if np.array_equal(depth_img, depth_uint16):
        print('encode_depth, decode_depth success!')