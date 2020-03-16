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


def decode_depth(depth, order='RGB'):  # R(h8) G(l8) B(c)  --> uint16
    assert order in ['RGB', 'BGR']
    if order == 'RGB':
        idx_h, idx_l, idx_c = 0, 1, 2
    else:  # order == 'BGR':
        idx_h, idx_l, idx_c = 2, 1, 0
    h8 = depth[:, :, idx_h].astype(np.uint16)  # high 8 bits
    l8 = depth[:, :, idx_l].astype(np.uint16)  # low 8 bits
    # c = depth[:, :, idx_c].astype(np.uint16)  # empty
    depth_uint16 = (h8 << 8) + l8
    # print('bin high 8', format(h8[300, 300], 'b'))
    # print('bin low  8', format(l8[300, 300], 'b'))
    # print('bin uint16', format(depth_uint16[300, 300], 'b'))
    return depth_uint16
