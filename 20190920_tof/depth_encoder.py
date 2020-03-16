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

    rgb_filename = '/home/reid/20190920_tof/100_rgb.png'
    rgb_img = cv2.imread(rgb_filename)
    print(rgb_img.shape)
    cv2.line(rgb_img,(252,259),(5,378),(0,0,255),2)
    cv2.line(rgb_img,(637,381),(371,265),(0,0,255),2)
    cv2.imshow('rgb', rgb_img)
    cv2.waitKey()

    depth_filename = '/home/reid/20190920_tof/100_dpt.png'
    # depth_img = imageio.imread(depth_filename)  # <class 'numpy.uint16'>
    depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    print(depth_img.shape)
    depth_3uint8 = encode_depth(depth_img)
    cv2.line(depth_3uint8,(252,259),(5,378),(0,0,255),2)
    cv2.line(depth_3uint8,(637,381),(371,265),(0,0,255),2)
    cv2.imshow('depth_3uint8', depth_3uint8)
    cv2.waitKey()
    print(depth_3uint8.shape)
    # print('0 ', np.max(depth_3uint8[:, :, 0]))
    # print('1 ', np.max(depth_3uint8[:, :, 1]))
    # print('2 ', np.max(depth_3uint8[:, :, 2]))
    # imageio.imwrite('name_mask.jpg', depth_3uint8)
	#depth_8bit_3channel = encode_depth(depth_16bit)
    img_out = np.concatenate((depth_3uint8, rgb_img), axis=1)  # (H, W1+W2, C)

    cv2.imshow('img_out', img_out)
    cv2.waitKey()
    #cv2.imwrite('name_mask_cv2.jpg', depth_3uint8)

    cv2.imwrite('name_mask_cv2_1.jpg', img_out)

    depth_uint16 = decode_depth(depth_3uint8)
    if np.array_equal(depth_img, depth_uint16):
        print('encode_depth, decode_depth success!')
