#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LI.Jie@Kiktech
# Feb. 20, 2020

import numpy as np


def get_dist_in_mask(depth, mask):
    idx_pixels = np.array(np.where(mask == 1))
    depth_values_in_mask = depth[idx_pixels[0, :], idx_pixels[1, :]]
    return np.mean(depth_values_in_mask)


#
# region: (p1(x0,y0), p2(x1,y1))
# -----------------------------------> X
# |          p1-------------
# |           |            |
# |           |            |
# |           |            |
# |           |            |
# |           |            |
# |           -------------p2
# |
# v
#  Y
def get_dist_in_region(depth, region):
    x0, y0 = region[0]
    x1, y1 = region[1]
    depth_values_in_region = depth[y0:y1, x0:x1]
    return np.mean(depth_values_in_region)


# boundingbox: (p1(x0,y0), p2(x1,y1)), 定义同上
def mark_image_boundingbox(img, boundingbox, color=(0, 255, 0)):
    x0, y0 = boundingbox[0]
    x1, y1 = boundingbox[1]
    # img[boundingbox[0]:boundingbox[2], boundingbox[1], :] = color
    # img[boundingbox[0]:boundingbox[2], boundingbox[3], :] = color
    # img[boundingbox[0], boundingbox[1]:boundingbox[3], :] = color
    # img[boundingbox[2], boundingbox[1]:boundingbox[3], :] = color
    img[y0:y1, x0, :] = color
    img[y0:y1, x1, :] = color
    img[y0, x0:x1, :] = color
    img[y1, x0:x1, :] = color
    return img
