import numpy as np
'''
bbox 转3D块
'''
def GetNoUseBbox(coordinate_list_fixed,flag):
    '''
    获取 flag中 未用过的所有bbox
    '''
    pass


def CalculatedDistance(current_bbox,coordinate_no_use):
    '''
    计算 传入bbox 与  flag中未用过的bbox的距离
    '''
    pass


def MergeBbox(i,j,distance,distance_threshold,flag):
    '''
    合并bbox,之后将 使用的bbox 在flag对应位置标注为 已使用

    '''
    # return merge_result,update_flag

def transform(merge_result,coordinate_list_fixed):
    '''
    将合并结果转化为3D坐标
    '''

def test():
    ##########################################################################
    list_file = 'a_bounding_box.txt'  # 单个序列坐标的文件
    distance_threshold = 5  #将中心点距离为5mm以内的bbox合并
    ##########################################################################
    with open(list_file) as f:
        lines = f.readlines()

    coordinate_list = []

    coordinate_len_list = []
    for line in lines:
        coordinate = line.split(',')
        coordinate.pop(0)  # 去掉 序列图像的下标值
        coordinate.pop(-1)  # 去掉 换行符
        coordinate_list.append(np.array(list(map(int, coordinate))))
        coordinate_len_list.append(len(coordinate))
    longest_len = max(coordinate_len_list) / 4  # 拿到 序列图中bbox最多的 框数量

    flag = np.zeros((len(coordinate_len_list), int(longest_len)))  # 序列中每个bbox的标识符，0 未使用，1 已使用

    coordinate_list_fixed = coordinate_list.copy()  # 所有bbox

    node_result=[] # 保存3D坐标的所有结果
    # 遍历所有图像
    for i in range(len(coordinate_list)):
        # 遍历一张图像的所有bbox
        for j in range(int(len(coordinate_list[i])/4)): # 步长为4
            # 计算每个bbox与 所有bbox的IOU or 中心点距离

            # 1、先判断flag中 当前bbox 是否已被使： 0 表示未使用  1：表示已使用
            if flag[i][j] == 1:
                # 已使用，查看下一个框
                break
            else:
                # 未使用,计算 其 与 flag中 未用过的所有bbox的距离

                # 当前bbox
                current_bbox=coordinate_list[i][j*4:j*4+4]
                # 获取 flag中 未用过的所有bbox
                coordinate_no_use=GetNoUseBbox(coordinate_list_fixed,flag)

                # 计算 传入bbox 与  flag中未用过的bbox的距离
                distance = CalculatedDistance(current_bbox,coordinate_no_use)

                # 合并bbox
                merge_result,flag=MergeBbox(i,j,distance,distance_threshold,flag)

                # 将合并结果转化为3D坐标
                node_3d = transform(merge_result,coordinate_list_fixed)


                # 保存淋巴结3D坐标
                node_result.append(node_3d)
    return node_result
if __name__ == '__main__':
    test()