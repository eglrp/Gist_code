import numpy as np
'''
一整个序列中  两个3D块IOU计算,但是不可取？可以考虑  总体积占真实淋巴结块的比例
'''
eps = 0.0001
def calcArea(box_one, box_two):

    xmin_, ymin_, xmax_, ymax_ = box_one
    xmin, ymin, xmax, ymax =box_two
    # xmin_, ymin_, xmax_, ymax_ = int(xmin_), int(ymin_), int(xmax_), int(ymax_)
    # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    # 判断
    if xmax_ < xmin_ or ymax_ < ymin_:
        print('box_one 坐标有问题===============')
    if xmax < xmin or ymax < ymin:
        print('box_two 坐标有问题===============')



    one_x, one_y, one_w, one_h = int((xmin_ + xmax_) / 2), int((ymin_ + ymax_) / 2), xmax_ - xmin_, ymax_ - ymin_
    two_x, two_y, two_w, two_h = int((xmin + xmax) / 2), int((ymin + ymax) / 2), xmax - xmin, ymax - ymin

    if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))
        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))
        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)
        inter_square = inter_w * inter_h

    return inter_square




# 参数：中心点x,y,z,x_len,y_len,z_len
cube_A = [3/2,3/2,3/2,3,3,3]
cube_B = [2,2,2,4,4,4]

# 遍历z，然后看每一层z的IOU
overlap_area_list=[]
# 在当前z层上，当两个块在该层都有值时，计算iou
A_x_min, A_y_min, A_z_min, A_x_max, A_y_max, A_z_max = cube_A[0] - cube_A[3] / 2, cube_A[1] - cube_A[4] / 2, cube_A[2] - \
                                                       cube_A[5] / 2, cube_A[0] + cube_A[3] / 2, cube_A[1] + cube_A[
                                                           4] / 2, cube_A[2] + cube_A[5] / 2
B_x_min, B_y_min, B_z_min, B_x_max, B_y_max, B_z_max = cube_B[0] - cube_B[3] / 2, cube_B[1] - cube_B[4] / 2, cube_B[2] - \
                                                       cube_B[5] / 2, cube_B[0] + cube_B[3] / 2, cube_B[1] + cube_B[
                                                           4] / 2, cube_B[2] + cube_B[5] / 2

# 体积
A_volume = (A_x_max - A_x_min) * (A_y_max - A_y_min) * (A_z_max - A_z_min)
B_volume = (B_x_max - B_x_min) * (B_y_max - B_y_min) * (B_z_max - B_z_min)


for i in range(int(max([A_z_max,B_z_max]))+1):
    # 重叠面积
    if A_z_min <= i < A_z_max and B_z_min <= i < B_z_max:
        overlap_area = calcArea([A_x_min,A_y_min,A_x_max,A_y_max], [B_x_min, B_y_min,B_x_max, B_y_max])  # 重叠面积
        overlap_area_list.append(overlap_area)

Inter_AB=sum(overlap_area_list) # 交集体积
Union_AB=A_volume+B_volume-Inter_AB

iou_volume=Inter_AB/(Union_AB+eps)
print(iou_volume)  #0.42
