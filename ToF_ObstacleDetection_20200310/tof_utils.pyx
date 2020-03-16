import numpy as np
cimport numpy as np
from cpython cimport array
# cimport cpython

def c_tof_template_match(grid, floor_y, mask):
    # ---- get shape of the 3D grid
    size_grid = grid.shape

    # ---- get shape of the mask
    size_mask = mask.shape

    # ----
    cdef np.ndarray[np.uint8_t, ndim=3] binary_voxel
    cdef np.ndarray[np.uint8_t, ndim=3] cue
    cdef np.ndarray[np.uint8_t, ndim=3] arr_leg_mask

    cdef np.ndarray[np.int32_t, ndim=3] c_grid
    cdef int sg_x, sg_y, sg_z  # shape of the 3D grid
    cdef int sm_x, sm_y, sm_z  # shape of the mask and cue
    cdef int _i_x, _i_y, _i_z
    cdef int sm_x_s, sm_x_e, sm_y_s, sm_y_e, sm_z_s, sm_z_e

    cdef int c_cy

    # --- shape of the 3D grid
    sg_x = size_grid[0]
    sg_y = size_grid[1]
    sg_z = size_grid[2]

    # shape of the mask and cue
    sm_x = size_mask[0]
    sm_y = size_mask[1]
    sm_z = size_mask[2]

    sm_x_s = sm_x//2
    sm_y_s = sm_y//2
    sm_z_s = sm_z//2
    sm_x_e = sm_x - sm_x_s
    sm_y_e = sm_y - sm_y_s
    sm_z_e = sm_z - sm_z_s

    # ----
    c_grid = grid
    # ----
    arr_leg_mask = mask
    # ---- center of the pallet leg, above the floor
    c_cy = floor_y - sm_y_s

    binary_voxel = np.zeros(size_grid, dtype=np.uint8)
    binary_voxel[c_grid > 0] = 1

    score_list = list()
    for _i_x in range(12, sg_x - 12):  # 15% --- 85%
        # for _i_y in range(c_cy - 5, c_cy - 1):
        for _i_y in range(c_cy - sm_y_s, c_cy + sm_y_e):  #
            for _i_z in range(41, sg_z):  # 410 mm ----- to the furthest voxel
                if c_grid[_i_x, _i_y, _i_z] > 0:
                    # ---- size of 'cue' must be the same with the 'mask'
                    # cue = binary_voxel[_i_x - 2:_i_x + 2, _i_y - 5:_i_y + 5, _i_z - 1:_i_z + 1]
                    cue = binary_voxel[_i_x - sm_x_s:_i_x + sm_x_e, _i_y - sm_y_s:_i_y + sm_y_e, _i_z - sm_z_s:_i_z + sm_z_e]
                    score_mask = np.multiply(cue, arr_leg_mask)
                    # The closer to the center of X, the higher score.
                    final_score = np.sum(score_mask) * (1 - abs(_i_x - 0.5 * sg_x) / sg_x)
                    # final_score = np.sum(score_mask) * math.pow((1 - abs(_i_x - size[0]/2)/size[0]), 3)
                    score_list.append((_i_x, _i_y, _i_z, final_score))

    score_array = np.asarray(score_list)
    idx_list = np.argmax(score_array[:, 3])
    tx = score_list[idx_list][0]
    ty = score_list[idx_list][1]
    tz = score_list[idx_list][2]
    return tx, ty, tz

def c_tof_depth2voxel(depth, unit, gridsize, offset, cam_k):
    # cdef array.array voxel_size = array.array("i", (80, 120, 140))
    # cdef np.ndarray[np.float32_t, ndim=2] cam_k
    cdef np.ndarray[np.float32_t, ndim=3] pt_cam
    cdef np.ndarray[np.int32_t, ndim=3] index_depth2grid
    # cdef np.ndarray[np.int32_t, ndim=3] voxel_binary
    cdef np.ndarray[np.int32_t, ndim=3] voxel_count
    # cdef np.ndarray[np.int32_t, ndim=2] position  #
    cdef np.ndarray[np.int32_t, ndim=4] index_grid2depth

    cdef int s_x, s_y, s_z
    cdef int i_x, i_y, i_z

    cdef int img_h, img_w
    img_h = depth.shape[0]
    img_w = depth.shape[1]

    # voxel_size = (80, 120, 140)  # TODO
    s_x = gridsize[0]
    s_y = gridsize[1]
    s_z = gridsize[2]

    offset_x = offset[0]
    offset_y = offset[1]
    offset_z = offset[2]

    # ---- Get point in camera coordinate
    gx, gy = np.meshgrid(range(img_w), range(img_h))

    pt_cam = np.zeros((img_h, img_w, 3), dtype=np.float32)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0] + offset_x  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1] + offset_y  # y
    pt_cam[:, :, 2] = depth + offset_z  # z, in meter

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

    del depth, gx, gy, pt_cam     # Release Memory
    # return voxel_binary, point_grid, voxel_hw_index   # (img_w, img_h, D), (img_w, img_h, D, 3)
    return voxel_count, index_depth2grid, index_grid2depth   # (img_w, img_h, D), (img_w, img_h, D, 3)
