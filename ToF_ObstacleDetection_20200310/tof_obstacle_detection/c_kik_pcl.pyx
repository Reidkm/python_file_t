# LI.Jie@Kiktech
# Jan. 19, 2020

import numpy as np
cimport numpy as np


# ---- 模板匹配
def template_match(grid, mask, anchor='CENTER'):
    cdef np.ndarray[np.uint8_t, ndim=3] binary_voxel
    cdef np.ndarray[np.uint8_t, ndim=3] cue

    cdef np.ndarray[np.int32_t, ndim=3] c_grid
    cdef int sg_x, sg_y, sg_z  # shape of the 3D grid
    cdef int sm_x, sm_y, sm_z
    cdef int sm_x_s, sm_x_e, sm_y_s, sm_y_e, sm_z_s, sm_z_e

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

    # ---- TODO: search region
    sr_x_s = 100  # x[0, 300], 100 <----> 200
    sr_x_e = 200
    sr_y_s = 0  # y[0, 150], y==20height==0
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

    # template match: using the center of the template as the anchor point
    cdef float best_score, final_score
    best_score = -1.0
    cdef int _i_x, _i_y, _i_z
    if anchor == 'CENTER':
        for _i_x in range(sr_x_s, sr_x_e):
            for _i_y in range(sr_y_s, sr_y_e):
                for _i_z in range(sr_z_s, sr_z_e):
                    if c_grid[_i_x, _i_y, _i_z] > 0:
                        # ---- size of 'cue' must be the same with the 'mask'
                        # cue = binary_voxel[_i_x - 2:_i_x + 2, _i_y - 5:_i_y + 5, _i_z - 1:_i_z + 1]
                        cue = binary_voxel[_i_x - sm_x_s:_i_x + sm_x_e, _i_y - sm_y_s:_i_y + sm_y_e, _i_z - sm_z_s:_i_z + sm_z_e]
                        # score_mask = np.multiply(cue, grid_mask)  # element-wise product
                        score_mask = cue * grid_mask  # element-wise product
                        # The closer to the center of X, the higher score.
                        final_score = np.sum(score_mask) * (1 - abs(_i_x - 0.5 * sg_x) / sg_x)
                        # final_score = np.sum(score_mask) * math.pow((1 - abs(_i_x - size[0]/2)/size[0]), 3)
                        #
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


# ---- 将深度图转换为点云
def get_point_cloud(depth, cam_k):
    # ---- Get point in camera coordinate
    cdef np.ndarray[np.float32_t, ndim=3] pt_cam
    cdef np.ndarray[np.int32_t, ndim=2] gx
    cdef np.ndarray[np.int32_t, ndim=2] gy

    cdef int img_h, img_w
    img_h = depth.shape[0]
    img_w = depth.shape[1]

    # print(depth.shape)
    gx, gy = np.meshgrid(range(img_w), range(img_h))
    pt_cam = np.zeros((img_h, img_w, 3), dtype=np.float32)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
    pt_cam[:, :, 2] = depth  # z, in meter

    # cdef np.ndarray[np.int32_t, ndim=2] c_depth
    # cdef np.ndarray[np.float32_t, ndim=2] c_cam_k
    # c_depth = depth
    # c_cam_k = cam_k
    # for h in range(img_h):
    #     for w in range(img_w):
    #         pt_cam[h][w][0] = (gx[h][w] - c_cam_k[0][2]) * c_depth[h][w] / c_cam_k[0][0]  # x
    #         pt_cam[h][w][1] = (gy[h][w] - c_cam_k[1][2]) * c_depth[h][w] / c_cam_k[1][1]  # y
    #         pt_cam[h][w][2] = c_depth[h][w]  # z, in meter

    del gx, gy,
    return pt_cam


# ---- 后续会删除该函数
def depth2voxel(depth, unit, gridsize, offset, cam_k):
    cdef np.ndarray[np.float32_t, ndim=3] pt_cam
    cdef np.ndarray[np.int32_t, ndim=3] index_depth2grid
    cdef np.ndarray[np.int32_t, ndim=3] voxel_count
    cdef np.ndarray[np.int32_t, ndim=4] index_grid2depth

    cdef int s_x, s_y, s_z
    cdef int i_x, i_y, i_z

    cdef int img_h, img_w
    img_h = depth.shape[0]
    img_w = depth.shape[1]

    s_x = gridsize[0]
    s_y = gridsize[1]
    s_z = gridsize[2]

    # ---- Get point in camera coordinate
    pt_cam = get_point_cloud(depth, cam_k)
    pt_cam += offset

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


# --- 找最大内接矩形
def findMaxRect(data):
    '''http://stackoverflow.com/a/30418912/5008845'''
    # print('data.shape', data.shape)
    cdef int nrows, ncols
    nrows = data.shape[0]
    ncols = data.shape[1]

    cdef np.ndarray[np.int32_t, ndim=2] h = np.zeros(dtype=np.int32, shape=data.shape)
    cdef np.ndarray[np.int32_t, ndim=2] w = np.zeros(dtype=np.int32, shape=data.shape)

    # area_max = (0, [])
    cdef float area_max = 0
    cdef float area
    cdef int minw
    cdef int skip = 1

    cdef int r, c, dh
    cdef int ss1, ss2, ss3, ss4
    for r in range(nrows):
        for c in range(ncols):
            if data[r, c] == skip:
                continue
            if r == 0:
                h[r, c] = 1
            else:
                h[r, c] = h[r - 1, c] + 1
            if c == 0:
                w[r, c] = 1
            else:
                w[r, c] = w[r, c - 1] + 1
            minw = w[r, c]
            for dh in range(h[r, c]):
                minw = min(minw, w[r - dh, c])
                # minw = minw if minw < w[r - dh][c] else w[r - dh][c]
                area = (dh + 1) * minw
                # if area > area_max[0]:
                #     area_max = (area, [(r - dh, c - minw + 1, r, c)])
                if area > area_max:
                    area_max = area
                    ss1 = r - dh
                    ss2 = c - minw  + 1
                    ss3 = r
                    ss4 = c
    # print(area, r - dh, c - minw + 1, r, c)
    # return area_max
    return (area_max, [(ss1, ss2, ss3, ss4)])

# ---- 点云转体素
def point_cloud2voxel(pt_cam, unit, gridsize, offset):
    # cdef array.array voxel_size = array.array("i", (80, 120, 140))
    # cdef np.ndarray[np.float32_t, ndim=2] cam_k
    # cdef np.ndarray[np.float32_t, ndim=3] pt_cam
    cdef np.ndarray[np.int32_t, ndim=3] index_depth2grid
    # cdef np.ndarray[np.int32_t, ndim=3] voxel_binary
    cdef np.ndarray[np.int32_t, ndim=3] voxel_count
    # cdef np.ndarray[np.int32_t, ndim=2] position  #
    cdef np.ndarray[np.int32_t, ndim=4] index_grid2depth

    cdef int s_x, s_y, s_z
    cdef int i_x, i_y, i_z

    cdef int img_h, img_w
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
