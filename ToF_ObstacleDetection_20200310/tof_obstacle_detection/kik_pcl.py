#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LI.Jie@Kiktech
# Jan. 10, 2020

import numpy as np


def findMaxRect(data):
    '''http://stackoverflow.com/a/30418912/5008845'''
    # print('data.shape', data.shape)
    nrows, ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    skip = 1
    area_max = (0, [])

    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [(r - dh, c - minw + 1, r, c)])
    # print(area, r - dh, c - minw + 1, r, c)
    return area_max


def template_match_del(grid, mask, anchor='CENTER'):
    # ---- get shape of the 3D grid
    size_grid = grid.shape
    # --- shape of the 3D grid
    sg_x = size_grid[0]
    sg_y = size_grid[1]
    sg_z = size_grid[2]

    # ---- get shape of the mask
    size_mask = mask.shape
    # ---- shape of the mask and cue
    sm_x = size_mask[0]
    sm_y = size_mask[1]
    sm_z = size_mask[2]

    # template mask
    sm_x_s = sm_x//2
    sm_y_s = sm_y//2
    sm_z_s = sm_z//2
    sm_x_e = sm_x - sm_x_s
    sm_y_e = sm_y - sm_y_s
    sm_z_e = sm_z - sm_z_s

    # ---- 确定搜索范围
    # search region, TODO:改为由相对与相机中心的物理距离来设定
    sr_x_s = 100  # x[0, 300], 100 <----> 200
    sr_x_e = 200
    sr_y_s = 0  # y[0, 150], y==20对应的height==0
    sr_y_e = 40
    sr_z_s = 100  # z[0, 300]
    sr_z_e = 150
    # print(sm_x_s, sm_y_s, sm_z_s)
    # print(sm_x_e, sm_y_e, sm_z_e)

    # sg_crop_rz = max(sm_z_s, sm_z_e)
    # ----
    c_grid = grid
    # ----
    grid_mask = mask
    # ---- search center of the pallet
    # c_cy = floor_y - sm_y_s

    binary_voxel = np.zeros(size_grid, dtype=np.uint8)
    binary_voxel[c_grid > 0] = 1

    score_list = list()

    # template match: using the center of the template as the anchor point
    if anchor == 'CENTER':
        for _i_x in range(sr_x_s, sr_x_e):
            for _i_y in range(sr_y_s, sr_y_e):
                for _i_z in range(sr_z_s, sr_z_e):
                    if c_grid[_i_x, _i_y, _i_z] > 0:
                        # ---- size of 'cue' must be the same with the 'mask'
                        # cue = binary_voxel[_i_x - 2:_i_x + 2, _i_y - 5:_i_y + 5, _i_z - 1:_i_z + 1]
                        cue = binary_voxel[_i_x - sm_x_s:_i_x + sm_x_e, _i_y - sm_y_s:_i_y + sm_y_e, _i_z - sm_z_s:_i_z + sm_z_e]
                        score_mask = np.multiply(cue, grid_mask)
                        # The closer to the center of X, the higher score.
                        final_score = np.sum(score_mask) * (1 - abs(_i_x - 0.5 * sg_x) / sg_x)
                        # final_score = np.sum(score_mask) * math.pow((1 - abs(_i_x - size[0]/2)/size[0]), 3)
                        score_list.append((_i_x, _i_y, _i_z, final_score))

    # template match: using the left-top corner of the template as the anchor point
    # if anchor == 'LEFT-TOP':
    #     for _i_x in range(12, sg_x - 12 - sm_x):  # 15% --- 85%
    #         for _i_y in range(c_cy - sm_y, c_cy + sm_y):  #
    #             for _i_z in range(41, sg_z - sm_z):  # 410 mm ----- to the furthest voxel
    #                 if c_grid[_i_x, _i_y, _i_z] > 0:
    #                     # ---- size of 'cue' must be the same with the 'mask'
    #                     cue = binary_voxel[_i_x:_i_x + sm_x, _i_y:_i_y + sm_y, _i_z:_i_z + sm_z]
    #                     score_mask = np.multiply(cue, grid_mask)
    #                     # The closer to the center of X, the higher score.
    #                     final_score = np.sum(score_mask) * (1 - abs(_i_x - 0.5 * sg_x) / sg_x)
    #                     # final_score = np.sum(score_mask) * math.pow((1 - abs(_i_x - size[0]/2)/size[0]), 3)
    #                     score_list.append((_i_x, _i_y, _i_z, final_score))

    if len(score_list) == 0:
        print("x: {}--{}".format(sr_x_s, sr_x_e))
        print("y: {}--{}".format(sr_y_s, sr_y_e))
        print("z: {}--{}".format(sr_z_s, sr_z_e))
        raise Exception('Template Matching failed. Anchor type is {}.'.format(anchor))

    score_array = np.asarray(score_list)
    idx_list = np.argmax(score_array[:, 3])
    tx = score_list[idx_list][0]
    ty = score_list[idx_list][1]
    tz = score_list[idx_list][2]
    return tx, ty, tz


def template_match(grid, mask, anchor='CENTER'):
    # ---- get shape of the 3D grid
    size_grid = grid.shape
    # --- shape of the 3D grid
    sg_x = size_grid[0]
    sg_y = size_grid[1]
    sg_z = size_grid[2]

    # ---- get shape of the mask
    size_mask = mask.shape
    # ---- shape of the mask and cue
    sm_x = size_mask[0]
    sm_y = size_mask[1]
    sm_z = size_mask[2]

    # template mask
    sm_x_s = sm_x//2
    sm_y_s = sm_y//2
    sm_z_s = sm_z//2
    sm_x_e = sm_x - sm_x_s
    sm_y_e = sm_y - sm_y_s
    sm_z_e = sm_z - sm_z_s

    # ---- 确定搜索范围
    # search region, TODO:改为由相对与相机中心的物理距离来设定
    sr_x_s = 100  # x[0, 300], 100 <----> 200
    sr_x_e = 200
    sr_y_s = 0  # y[0, 150], y==20对应的height==0
    sr_y_e = 40
    sr_z_s = 100  # z[0, 300]
    sr_z_e = 150
    # print(sm_x_s, sm_y_s, sm_z_s)
    # print(sm_x_e, sm_y_e, sm_z_e)

    # sg_crop_rz = max(sm_z_s, sm_z_e)
    # ----
    c_grid = grid
    # ----
    grid_mask = mask
    # ---- search center of the pallet
    # c_cy = floor_y - sm_y_s

    binary_voxel = np.zeros(size_grid, dtype=np.uint8)
    binary_voxel[c_grid > 0] = 1

    # score_list = list()

    best_score = -1.0
    # template match: using the center of the template as the anchor point
    if anchor == 'CENTER':
        for _i_x in range(sr_x_s, sr_x_e):
            for _i_y in range(sr_y_s, sr_y_e):
                for _i_z in range(sr_z_s, sr_z_e):
                    if c_grid[_i_x, _i_y, _i_z] > 0:
                        # ---- size of 'cue' must be the same with the 'mask'
                        # cue = binary_voxel[_i_x - 2:_i_x + 2, _i_y - 5:_i_y + 5, _i_z - 1:_i_z + 1]
                        cue = binary_voxel[_i_x - sm_x_s:_i_x + sm_x_e, _i_y - sm_y_s:_i_y + sm_y_e, _i_z - sm_z_s:_i_z + sm_z_e]
                        score_mask = np.multiply(cue, grid_mask)
                        # The closer to the center of X, the higher score.
                        final_score = np.sum(score_mask) * (1 - abs(_i_x - 0.5 * sg_x) / sg_x)
                        # final_score = np.sum(score_mask) * math.pow((1 - abs(_i_x - size[0]/2)/size[0]), 3)
                        # score_list.append((_i_x, _i_y, _i_z, final_score))
                        if final_score > best_score:
                            best_score = final_score
                            tx, ty, tz = _i_x, _i_y, _i_z

    # template match: using the left-top corner of the template as the anchor point
    # if anchor == 'LEFT-TOP':
    #     for _i_x in range(12, sg_x - 12 - sm_x):  # 15% --- 85%
    #         for _i_y in range(c_cy - sm_y, c_cy + sm_y):  #
    #             for _i_z in range(41, sg_z - sm_z):  # 410 mm ----- to the furthest voxel
    #                 if c_grid[_i_x, _i_y, _i_z] > 0:
    #                     # ---- size of 'cue' must be the same with the 'mask'
    #                     cue = binary_voxel[_i_x:_i_x + sm_x, _i_y:_i_y + sm_y, _i_z:_i_z + sm_z]
    #                     score_mask = np.multiply(cue, grid_mask)
    #                     # The closer to the center of X, the higher score.
    #                     final_score = np.sum(score_mask) * (1 - abs(_i_x - 0.5 * sg_x) / sg_x)
    #                     # final_score = np.sum(score_mask) * math.pow((1 - abs(_i_x - size[0]/2)/size[0]), 3)
    #                     score_list.append((_i_x, _i_y, _i_z, final_score))

    # if len(score_list) == 0:
    if best_score <= 0:
        print("x: {}--{}".format(sr_x_s, sr_x_e))
        print("y: {}--{}".format(sr_y_s, sr_y_e))
        print("z: {}--{}".format(sr_z_s, sr_z_e))
        raise Exception('Template Matching failed. Anchor type is {}.'.format(anchor))

    # score_array = np.asarray(score_list)
    # idx_list = np.argmax(score_array[:, 3])
    # tx = score_list[idx_list][0]
    # ty = score_list[idx_list][1]
    # tz = score_list[idx_list][2]
    return tx, ty, tz


def get_point_cloud(depth, cam_k):
    # ---- Get point in camera coordinate
    H, W = depth.shape
    # print(depth.shape)
    gx, gy = np.meshgrid(range(W), range(H))
    pt_cam = np.zeros((H, W, 3), dtype=np.float32)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
    pt_cam[:, :, 2] = depth  # z, in meter
    # print(gx.shape)
    # print(cam_k.shape)
    # print(pt_cam.shape)

    # img_h = depth.shape[0]
    # img_w = depth.shape[1]
    # for h in range(img_h):
    #     for w in range(img_w):
    #         pt_cam[h, w, 0] = (gx[h, w] - cam_k[0][2]) * depth[h, w] / cam_k[0][0]  # x
    #         pt_cam[h, w, 1] = (gy[h, w] - cam_k[1][2]) * depth[h, w] / cam_k[1][1]  # y
    #         pt_cam[h, w, 2] = depth[h, w]  # z, in meter

    del gx, gy,
    return pt_cam


def get_color_point_cloud(rgb, depth, cam_k):
    pt_cam = get_point_cloud(depth, cam_k)
    return np.concatenate((pt_cam, rgb[:, :, 0:3]), axis=2)  # H x W x 6


def save_point_cloud(point_clouds, ply_filename):
    """ Save point clouds (x, y, z) or color point coluds (x, y, z, r, g, b) 
    """  #
    if isinstance(point_clouds, list):
        points_number = len(point_clouds)
        n = len(point_clouds[0])
        ply_data = point_clouds
    else:  # array
        # Shape of point_clouds is H x W x n
        points_number = point_clouds.shape[0] * point_clouds.shape[1]
        n = point_clouds.shape[2]
        ply_data = point_clouds.reshape((-1, n))
    if n == 3:  # (x, y, z)
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'end_header' % points_number
        # ply_filename = filename + '_Depth2CPL.ply'
        np.savetxt(ply_filename, ply_data, fmt="%.2f %.2f %.2f", header=ply_head, comments='')
        print('Save the ply file to:{}.'.format(ply_filename))
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
        np.savetxt(ply_filename, ply_data, fmt="%.2f %.2f %.2f %d %d %d", header=ply_head, comments='')
        print('Save the ply file to:{}.'.format(ply_filename))
    else:
        print('save_point_cloud error.')
    # logger.info('Saved-->{}'.format(ply_filename))
    # print('Save the ply file to:{}.'.format(ply_filename))


# TODO：解释每个参数，并给出使用示例
# unit 每个体素的大小
# gridsize： 划分的栅格的数量
# offset:
def depth2voxel(depth, unit, gridsize, offset, cam_k):
    # cdef array.array voxel_size = array.array("i", (80, 120, 140))
    # cdef np.ndarray[np.float32_t, ndim=2] cam_k
    # cdef np.ndarray[np.float32_t, ndim=3] pt_cam
    # cdef np.ndarray[np.int32_t, ndim=3] index_depth2grid
    # cdef np.ndarray[np.int32_t, ndim=3] voxel_binary
    # cdef np.ndarray[np.int32_t, ndim=3] voxel_count
    # cdef np.ndarray[np.int32_t, ndim=2] position  #
    # cdef np.ndarray[np.int32_t, ndim=4] index_grid2depth

    # cdef int s_x, s_y, s_z
    # cdef int i_x, i_y, i_z

    # cdef int img_h, img_w
    img_h = depth.shape[0]
    img_w = depth.shape[1]

    # voxel_size = (80, 120, 140)  # TODO
    s_x = gridsize[0]
    s_y = gridsize[1]
    s_z = gridsize[2]

    # ---- Get point in camera coordinate
    pt_cam = get_point_cloud(depth, cam_k)

    pt_cam += offset
    # offset_x = offset[0]
    # offset_y = offset[1]
    # offset_z = offset[2]
    # gx, gy = np.meshgrid(range(img_w), range(img_h))
    # pt_cam = np.zeros((img_h, img_w, 3), dtype=np.float32)
    # pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0] + offset_x  # x
    # pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1] + offset_y  # y
    # pt_cam[:, :, 2] = depth + offset_z  # z, in meter

    index_depth2grid = np.rint(pt_cam / unit).astype(np.int32)
    # voxel_binary = np.zeros([_ + 1 for _ in voxel_size], dtype=np.int32)  # (img_w, img_h, D)
    # voxel_binary = np.zeros((s_x, s_y, s_z), dtype=np.int32)  # (img_w, img_h, D)
    voxel_count = np.zeros((s_x, s_y, s_z), dtype=np.int32)  # (img_w, img_h, D)
    index_grid2depth = np.zeros((s_x, s_y, s_z, 2), dtype=np.int32)  # (img_w, img_h, D, 2)

    for h in range(img_h):
        for w in range(img_w):
            # i_x, i_y, i_z = point_grid[h, w, :]  # python
            i_x = index_depth2grid[h, w, 0]  # cython
            i_y = index_depth2grid[h, w, 1]
            i_z = index_depth2grid[h, w, 2]
            if 0 <= i_x < s_x and 0 <= i_y < s_y and 0 <= i_z < s_z:
                # voxel_binary[i_x, i_y, i_z] = 1  # the bin has at least one point (bin is not empty)
                voxel_count[i_x, i_y, i_z] += 1  # the bin has at least one point (bin is not empty)
                # voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
                # position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_size)
                index_grid2depth[i_x, i_y, i_z, 0] = h
                index_grid2depth[i_x, i_y, i_z, 1] = w

    del depth, pt_cam     # Release Memory
    # return voxel_binary, point_grid, voxel_hw_index   # (img_w, img_h, D), (img_w, img_h, D, 3)
    return voxel_count, index_depth2grid, index_grid2depth   # (img_w, img_h, D), (img_w, img_h, D, 3)


def point_cloud2voxel(pt_cam, unit, gridsize, offset):
    # cdef array.array voxel_size = array.array("i", (80, 120, 140))
    # cdef np.ndarray[np.float32_t, ndim=2] cam_k
    # cdef np.ndarray[np.float32_t, ndim=3] pt_cam
    # cdef np.ndarray[np.int32_t, ndim=3] index_depth2grid
    # cdef np.ndarray[np.int32_t, ndim=3] voxel_binary
    # cdef np.ndarray[np.int32_t, ndim=3] voxel_count
    # cdef np.ndarray[np.int32_t, ndim=2] position  #
    # cdef np.ndarray[np.int32_t, ndim=4] index_grid2depth

    # cdef int s_x, s_y, s_z
    # cdef int i_x, i_y, i_z

    # cdef int img_h, img_w
    img_h = pt_cam.shape[0]
    img_w = pt_cam.shape[1]

    # voxel_size = (80, 120, 140)  # TODO
    s_x = gridsize[0]
    s_y = gridsize[1]
    s_z = gridsize[2]

    # ---- Get point in camera coordinate
    # pt_cam = get_point_cloud(depth, cam_k)

    pt_cam += offset
    # offset_x = offset[0]
    # offset_y = offset[1]
    # offset_z = offset[2]
    # gx, gy = np.meshgrid(range(img_w), range(img_h))
    # pt_cam = np.zeros((img_h, img_w, 3), dtype=np.float32)
    # pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0] + offset_x  # x
    # pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1] + offset_y  # y
    # pt_cam[:, :, 2] = depth + offset_z  # z, in meter

    index_depth2grid = np.rint(pt_cam / unit).astype(np.int32)
    
    # voxel_binary = np.zeros([_ + 1 for _ in voxel_size], dtype=np.int32)  # (img_w, img_h, D)
    # voxel_binary = np.zeros((s_x, s_y, s_z), dtype=np.int32)  # (img_w, img_h, D)
    voxel_count = np.zeros((s_x, s_y, s_z), dtype=np.int32)  # (img_w, img_h, D)
    index_grid2depth = np.zeros((s_x, s_y, s_z, 2), dtype=np.int32)  # (img_w, img_h, D, 2)

    for h in range(img_h):
        for w in range(img_w):
            # i_x, i_y, i_z = point_grid[h, w, :]  # python
            i_x = index_depth2grid[h, w, 0]  # cython
            i_y = index_depth2grid[h, w, 1]
            i_z = index_depth2grid[h, w, 2]
            if 0 <= i_x < s_x and 0 <= i_y < s_y and 0 <= i_z < s_z:
                # voxel_binary[i_x, i_y, i_z] = 1  # the bin has at least one point (bin is not empty)
                voxel_count[i_x, i_y, i_z] += 1  # the bin has at least one point (bin is not empty)
                # voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
                # position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_size)
                index_grid2depth[i_x, i_y, i_z, 0] = h
                index_grid2depth[i_x, i_y, i_z, 1] = w

    # del depth, pt_cam  # Release Memory
    del pt_cam  # Release Memory
    # return voxel_binary, point_grid, voxel_hw_index   # (img_w, img_h, D), (img_w, img_h, D, 3)
    return voxel_count, index_depth2grid, index_grid2depth  # (img_w, img_h, D), (img_w, img_h, D, 3)


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


# 将voxel中的部分区域保存
def save_region(voxel_val, pallet_size, pallet_center, npy_filename):
    smx, smy, smz = pallet_size
    tx, ty, tz = pallet_center  # center
    rx, ry, rz = smx - smx // 2, smy - smy // 2, smz - smz // 2
    # ----- save the pallet area
    np.save(npy_filename, voxel_val[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz])
    print("Save the pallet template to: {}.".format(npy_filename))


# 将voxel中的部分区域标记为特定的颜色，保存为彩色点云
def mark_certain_region_in_voxel(voxel_val, pallet_size, pallet_center):
    """ply: x y z of voxels from depth. only save the voxels containing points"""
    # ---- get size
    size = voxel_val.shape
    # print('size', size)
    _x, _y, _z = get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()

    ply_data_rgb = np.zeros(size + (3,), dtype=np.int32)

    # ----- 标记pallet所占的体素为RGB颜色(255, 0, 0)
    smx, smy, smz = pallet_size
    tx, ty, tz = pallet_center  # center
    rx, ry, rz = smx - smx // 2, smy - smy // 2, smz - smz // 2

    ply_data_rgb[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (255, 0, 0)

    # # ----- save the pallet area
    # np.save(npy_filename, voxel_val[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz])
    # print("Save the pallet template to: {}.".format(npy_filename))

    # ----- 标记pallet的X和Y轴的边界框为RGB颜色(0, 255, 0)
    # 向内部缩进2个格子
    ew = 2  # edge_width
    ply_data_rgb[tx - rx-ew:tx - rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_Left
    ply_data_rgb[tx + rx:tx + rx+ew, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_right
    ply_data_rgb[tx - rx:tx + rx, ty - ry:ty - ry+ew, tz - rz:tz + rz, :] = (0, 255, 0)  # y_bottom
    ply_data_rgb[tx - rx:tx + rx, ty + ry-ew:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # y_Left

    # -----------------------------------------------------------------
    # ----- 保存彩色点云
    r = ply_data_rgb[:, :, :, 0].flatten()
    g = ply_data_rgb[:, :, :, 1].flatten()
    b = ply_data_rgb[:, :, :, 2].flatten()

    ply_data_grid = zip(_x, _y, _z, r, g, b, voxel_val.flatten())
    ply_data_grid = list(ply_data_grid)

    ply_data = []
    for i_idx in range(len(ply_data_grid)):
        # if ply_data_grid[i_idx][6] > 3:  # 0 is empty, 1, only 1 point
        if ply_data_grid[i_idx][6] > 1:  # 0 is empty, 1, only 1 point
            # ply_data.append(ply_data_grid[i_idx])
            ply_data.append(ply_data_grid[i_idx][:-1])  # 去掉最后一个

    if len(ply_data) == 0:
        print('NO valid data.')
        return

    return ply_data


# 将voxel中的高于地面的区域标记为特定的颜色，保存为彩色点云
def mark_obstacle_in_voxel(voxel_val, ground, min_point_number=1):
    """ply: x y z of voxels from depth. only save the voxels containing points"""
    # ---- get size
    size = voxel_val.shape
    # print('size', size)
    _x, _y, _z = get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()

    ply_data_rgb = np.zeros(size + (3,), dtype=np.int32)

    # ----- 标记pallet所占的体素为RGB颜色(255, 0, 0)
    # smx, smy, smz = pallet_size
    # tx, ty, tz = pallet_center  # center
    # rx, ry, rz = smx - smx // 2, smy - smy // 2, smz - smz // 2

    # ply_data_rgb[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (255, 0, 0)

    # # ----- save the pallet area
    # np.save(npy_filename, voxel_val[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz])
    # print("Save the pallet template to: {}.".format(npy_filename))

    # ----- 标记pallet的X和Y轴的边界框为RGB颜色(0, 255, 0)
    # 向内部缩进2个格子
    ew = 2  # edge_width
    # ply_data_rgb[tx - rx-ew:tx - rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_Left
    # ply_data_rgb[tx + rx:tx + rx+ew, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_right
    # ply_data_rgb[tx - rx:tx + rx, ty - ry:ty - ry+ew, tz - rz:tz + rz, :] = (0, 255, 0)  # y_bottom
    # ply_data_rgb[tx - rx:tx + rx, ty + ry-ew:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # y_Left

    ply_data_rgb[:, ground:, :] = (255, 0, 0)

    # -----------------------------------------------------------------
    # ----- 保存彩色点云
    r = ply_data_rgb[:, :, :, 0].flatten()
    g = ply_data_rgb[:, :, :, 1].flatten()
    b = ply_data_rgb[:, :, :, 2].flatten()

    ply_data_grid = zip(_x, _y, _z, r, g, b, voxel_val.flatten())
    ply_data_grid = list(ply_data_grid)

    ply_data = []
    for i_idx in range(len(ply_data_grid)):
        if ply_data_grid[i_idx][6] >= min_point_number:  # 0 is empty, 1, only 1 point
            # ply_data.append(ply_data_grid[i_idx])
            ply_data.append(ply_data_grid[i_idx][:-1])  # 去掉最后一个

    if len(ply_data) == 0:
        print('NO valid data.')
        return

    return ply_data


# 将voxel中的高于地面的区域标记为特定的颜色，保存为彩色点云
def mark_detected_obstacle_in_voxel(voxel_val, ground, obstacles, sample_rate, min_point_number=1):
    """ply: x y z of voxels from depth. only save the voxels containing points"""
    # ---- get size
    size = voxel_val.shape
    # print('size', size)
    _x, _y, _z = get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()

    ply_data_rgb = np.zeros(size + (3,), dtype=np.int32)

    # ----- 标记pallet所占的体素为RGB颜色(255, 0, 0)
    # smx, smy, smz = pallet_size
    # tx, ty, tz = pallet_center  # center
    # rx, ry, rz = smx - smx // 2, smy - smy // 2, smz - smz // 2

    # ply_data_rgb[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (255, 0, 0)

    # # ----- save the pallet area
    # np.save(npy_filename, voxel_val[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz])
    # print("Save the pallet template to: {}.".format(npy_filename))

    # ----- 标记pallet的X和Y轴的边界框为RGB颜色(0, 255, 0)
    # 向内部缩进2个格子
    ew = 2  # edge_width
    # ply_data_rgb[tx - rx-ew:tx - rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_Left
    # ply_data_rgb[tx + rx:tx + rx+ew, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_right
    # ply_data_rgb[tx - rx:tx + rx, ty - ry:ty - ry+ew, tz - rz:tz + rz, :] = (0, 255, 0)  # y_bottom
    # ply_data_rgb[tx - rx:tx + rx, ty + ry-ew:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # y_Left

    # 标记检测区域：若原来为地面，则标记为颜色(0, 255, 0)
    ply_data_rgb[:, 0:ground+1, :] = (0, 255, 0)

    # 将高于地面的物体全部标记为障碍
    ply_data_rgb[:, ground+1:, :] = (255, 0, 0)

    # 标记障碍区域，若该区域存在障碍，则标记为 (255, 0, 255)
    # for idx_z in obstacles:
    #     if idx_z > 0:
    #         ply_data_rgb[:, 0, idx_z*sample_rate:(idx_z+1)*sample_rate] = (255, 0, 255)
    sr = sample_rate
    for _idx in range(obstacles.size):
        idx_z = obstacles[_idx]
        if idx_z > 0:
            # ply_data_rgb[_idx * sr:(_idx + 1) * sr, 0:ground, idx_z * sr:(idx_z + 1) * sr] = (255, 255, 255)
            # voxel_val[_idx * sr:(_idx + 1) * sr, 0:ground, idx_z * sr:(idx_z + 1) * sr] = min_point_number
            ply_data_rgb[_idx * sr:(_idx + 1) * sr, 0:ground+1, idx_z * sr:] = (255, 255, 255)
            voxel_val[_idx * sr:(_idx + 1) * sr, 0:ground+1, idx_z * sr:] = min_point_number

    # -----------------------------------------------------------------
    # ----- 保存彩色点云
    r = ply_data_rgb[:, :, :, 0].flatten()
    g = ply_data_rgb[:, :, :, 1].flatten()
    b = ply_data_rgb[:, :, :, 2].flatten()

    ply_data_grid = zip(_x, _y, _z, r, g, b, voxel_val.flatten())
    ply_data_grid = list(ply_data_grid)

    ply_data = []
    for i_idx in range(len(ply_data_grid)):
        if ply_data_grid[i_idx][6] >= min_point_number:  # 0 is empty, 1, only 1 point
            # ply_data.append(ply_data_grid[i_idx])
            ply_data.append(ply_data_grid[i_idx][:-1])  # 去掉最后一个
        # elif ply_data_grid[i_idx][4] == 255:  # 被标记为检测区域，对应地面体素为空,则修改颜色
        #     # ply_data_grid[i_idx][4] = 125  # ,'tuple' object does not support item assignment
        #     ply_data.append(ply_data_grid[i_idx][:-1])

    # -----划分栅格：
    # 水平线：
    # ply_grid_lines_x = np.zeros((size[0], size[2]//sample_rate, 3,), dtype=np.int32)
    size_xline = (size[0], 1, size[2]//sample_rate + 1)
    _x, _y, _z = get_xyz(size_xline)
    _z = _z * sample_rate - 0.5
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    ply_line_rgb = np.zeros(size_xline + (3,), dtype=np.int32)
    ply_line_rgb[:, 0, :] = (0, 0, 255)
    r = ply_line_rgb[:, :, :, 0].flatten()
    g = ply_line_rgb[:, :, :, 1].flatten()
    b = ply_line_rgb[:, :, :, 2].flatten()
    ply_line_grid = zip(_x, _y, _z, r, g, b)
    ply_line_grid = list(ply_line_grid)
    ply_data.extend(ply_line_grid)

    # 垂直线：
    # ply_grid_lines_z = np.zeros((size[0]//sample_rate, size[2], 3,), dtype=np.int32)
    size_xline = (size[0]//sample_rate + 1, 1, size[2])
    _x, _y, _z = get_xyz(size_xline)
    _x = _x * sample_rate - 0.5
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    ply_line_rgb = np.zeros(size_xline + (3,), dtype=np.int32)
    ply_line_rgb[:, 0, :] = (0, 0, 255)
    r = ply_line_rgb[:, :, :, 0].flatten()
    g = ply_line_rgb[:, :, :, 1].flatten()
    b = ply_line_rgb[:, :, :, 2].flatten()
    ply_line_grid = zip(_x, _y, _z, r, g, b)
    ply_line_grid = list(ply_line_grid)
    ply_data.extend(ply_line_grid)

    if len(ply_data) == 0:
        print('NO valid data.')
        return

    return ply_data


# TODO
def depth_voxel2ply(voxel_val, ply_filename):
    """ply: x y z of voxels from depth. only save the voxels containing points"""
    # ---- get size
    size = voxel_val.shape
    # print('size', size)
    _x, _y, _z = get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()

    # ply_data_grid = zip(_x, _y, _z, voxel_val.flatten())

    ply_data_rgb = np.zeros(size + (3,), dtype=np.int32)
    smx, smy, smz = 78, 12, 6  # mask.shape
    # center
    tx = 145
    ty = 32
    tz = 132
    # rx, ry, rz = smx//2, smy//2, smz//2
    rx, ry, rz = smx - smx // 2, smy - smy // 2, smz - smz // 2
    ply_data_rgb[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz, 0] = 255
    ply_data_rgb[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz, 1] = 0
    ply_data_rgb[tx - rx:tx + rx, ty - ry:ty + ry, tz - rz:tz + rz, 2] = 0

    # 向内部缩进2各格子
    ply_data_rgb[tx - rx-2:tx - rx, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_Left
    ply_data_rgb[tx + rx:tx + rx+2, ty - ry:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # x_right
    ply_data_rgb[tx - rx:tx + rx, ty - ry:ty - ry+2, tz - rz:tz + rz, :] = (0, 255, 0)  # y_bottom
    ply_data_rgb[tx - rx:tx + rx, ty + ry-2:ty + ry, tz - rz:tz + rz, :] = (0, 255, 0)  # y_Left

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

    # ply_data = []
    # for i_idx in range(len(ply_data_grid)):
    #     if ply_data_grid[i_idx][3] > 0:  # 0 is em
    #         ply_data.append(ply_data_grid[i_idx])

    if len(ply_data) == 0:
        print('From _depth_voxel2ply(): NO valid data. {}'.format(ply_filename))
        return
    # ply_head = 'ply\n' \
    #            'format ascii 1.0\n' \
    #            'element vertex %d\n' \
    #            'property float x\n' \
    #            'property float y\n' \
    #            'property float z\n' \
    #            'property int label\n' \
    #            'end_header' % len(ply_data)
    # np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d", header=ply_head, comments='')
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
