import numpy as np
'''
bbox 转3D块
'''
def list_2_numpy(coordinate_list):
    '''
    结果每行都为list，list里面4个为一组坐标
    coordinate_list  list[11,]
    '''
    result_list_2_numpy = []
    for coordinate in coordinate_list:
        temp_list_2_numpy=[]
        for i_list_2_numpy in range(0,len(coordinate),4):  # 步长为4
            temp_list_2_numpy.append([coordinate[i_list_2_numpy],coordinate[i_list_2_numpy+1],coordinate[i_list_2_numpy+2],coordinate[i_list_2_numpy+3]])
        result_list_2_numpy.append(temp_list_2_numpy)
    return result_list_2_numpy

def GetNoUseBbox(coordinate_list_grouping,flag):
    '''
    从coordinate_list_grouping获取 flag对应位置标记为0的 bbox
    coordinate_list_fixed：list[11,] 每行numpy长度不等
    '''
    # 遍历flag,取出对应位置标记为0的 coordinate_list_grouping的坐标块
    result_GetNoUseBbox=[]
    for i_GetNoUseBbox in range(len(flag)):
        temp_GetNoUseBbox=[]
        for j_GetNoUseBbox in range(flag[i_GetNoUseBbox].size):
            if flag[i_GetNoUseBbox][j_GetNoUseBbox] == 0:
                # 未使用
                temp_GetNoUseBbox.append(coordinate_list_grouping[i_GetNoUseBbox][j_GetNoUseBbox])
            else:
                # 已使用
                temp_GetNoUseBbox.append([0,0,0,0])
        result_GetNoUseBbox.append(temp_GetNoUseBbox)
    return result_GetNoUseBbox





def Calculated(current_bbox,coordinate_no_use):
    '''
    计算 传入bbox 与  flag中未用过的bbox  的 欧式距离（or  IOU）
    '''
    # 遍历coordinate_no_use    当一组坐标为[0,0,0,0],表示该坐标已被使用，距离设定为float('inf') 表示正无穷
    result_Calculated = []
    for i_coordinate_no_use in coordinate_no_use:
        temp_Calculated = []
        for j_i_coordinate_no_use in i_coordinate_no_use:

            if j_i_coordinate_no_use == [0,0,0,0]:
                # 标明该坐标已被使用
                temp_Calculated.append(float('inf'))
            else:
                # 计算欧式距离
                distance_Calculated = np.linalg.norm(np.array([((current_bbox[0]+current_bbox[2])/2),((current_bbox[1]+current_bbox[3])/2)])-np.array([((j_i_coordinate_no_use[0]+j_i_coordinate_no_use[2])/2),((j_i_coordinate_no_use[1]+j_i_coordinate_no_use[3])/2)]))
                temp_Calculated.append(distance_Calculated)
        result_Calculated.append(temp_Calculated)
    return result_Calculated



def MergeBbox(i,j,distance,distance_threshold,flag):
    '''
    合并bbox,之后将 使用的bbox 在flag对应位置标注为 已使用
    i，j:当前第i张的第j个bbox需要合并
    distance：该bbox与 所有未使用的bbox之间的距离
    distance_threshold：距离合并的阈值
    flag：确定合并bbox将 对应位置的flag置为1，即bbox标为已使用

    return:
            merge_result 形状与distance相同，仅有0、1值（1代表合并）
            flag  形状与flag相同，将 合并的bbox对应位置置为1（标为已使用）
    '''
    # 遍历distance
    result_MergeBbox= []
    # 刚开始result_MergeBbox全部置为0
    for i_MergeBbox in distance:
        result_MergeBbox.append(np.zeros((len(i_MergeBbox))).tolist())

    # 1、将i,j处置为1
    # 2、从i下一张开始判断，如果每行的最小距离小于等于阈值，则置为1
    result_MergeBbox[i][j]=1

    # 对应flag位置从0置为1
    if flag[i][j] == 1:
        print('合并出错1111')
    else:
        flag[i][j] == 1

    for i_2_MergeBbox in range(i+1,len(distance)): #下标值  从i+1开始
        if min(distance[i_2_MergeBbox])>distance_threshold:
            break  # 当出现整张不符合时，则 合并结束
        else:
            index=distance[i_2_MergeBbox].index(min(distance[i_2_MergeBbox])) #距离<=阈值的横轴下标值
            result_MergeBbox[i_2_MergeBbox][index]=1

            # 对应flag位置从0置为1
            if flag[i_2_MergeBbox][index] == 1:
                print('合并出错22222')
            else:
                flag[i_2_MergeBbox][index] == 1


    return result_MergeBbox,flag


def transform(merge_result,coordinate_list_fixed):
    '''
    将合并结果转化为3D坐标
    '''
    print()

def test():
    ##########################################################################
    list_file = 'a_bounding_box.txt'  # 单个序列坐标的文件
    distance_threshold = 6  #将中心点距离为5mm以内的bbox合并
    ##########################################################################
    with open(list_file) as f:
        lines = f.readlines()

    coordinate_list = []

    # 序列中每个bbox的标识符，0 未使用，1 已使用
    flag = []
    for line in lines:
        coordinate = line.split(',')
        coordinate.pop(0)  # 去掉 序列图像的下标值
        coordinate.pop(-1)  # 去掉 换行符
        coordinate_list.append(np.array(list(map(int, coordinate))))
        flag.append(np.zeros((int(len(coordinate)/4))))


    # 每行都为list，list里面4个为一组坐标
    coordinate_list_grouping=list_2_numpy(coordinate_list)


    node_result=[] # 保存3D坐标的所有结果
    # 遍历所有图像
    for i in range(len(coordinate_list)):
        # 遍历一张图像的所有bbox
        for j in range(int(len(coordinate_list[i])/4)):
            # 计算每个bbox与 所有bbox的IOU or 中心点距离

            # 1、先判断flag中 当前bbox 是否已被使： 0 表示未使用  1：表示已使用
            if flag[i][j] == 1:
                # 已使用，查看下一个框
                continue
            else:
                # 未使用,计算 其 与 flag中 未用过的所有bbox的距离

                # 当前bbox
                current_bbox=coordinate_list[i][j*4:j*4+4]
                # 获取 flag中 未用过的所有bbox
                coordinate_no_use=GetNoUseBbox(coordinate_list_grouping,flag)

                # 计算 传入bbox 与  flag中未用过的bbox的距离
                distance = Calculated(current_bbox,coordinate_no_use)

                # 合并bbox
                merge_result,flag=MergeBbox(i,j,distance,distance_threshold,flag)

                # 将合并结果转化为3D坐标
                node_3d = transform(merge_result,coordinate_list_grouping)


                # 保存淋巴结3D坐标
                node_result.append(node_3d)
    return node_result
if __name__ == '__main__':
    test()