import csv
import numpy as np
from config import opt
class getTruthCoordinate():
    '''
    得到所有 真值3D块坐标
    3D块坐标 标准（x_min,y_min,z_min,x_max,y_max,z_max）
    '''
    def __init__(self):
        self.truth_bounding_box_dir = opt.truth_bounding_box_dir

    def get_3d_coordinate(self):

        # 按照序列划分
        group_list = self.load_txt()

        result = []
        for single_list in group_list:
            # 真值 记录不连续，则转为连续
            single_list=self.transform(single_list)

            result.append(get_single_sequence(single_list).get_3d())

        return result

    def load_txt(self):
        """
        加载txt文件数据，并且将列表按病人分割为多个
        :param txt_dir: txt文件路径
        :return: 按病人分开的列表
        """
        with open(self.truth_bounding_box_dir, 'r') as bbox_txt:
            bbox_list = bbox_txt.readlines()
        bbox_list.pop(0)  # 去掉标题头
        pa_divide_list, temp = [], 0
        for i in range(len(bbox_list)):
            # 去掉最后的\n
            bbox_list[i] = bbox_list[i].rstrip('\n')
            # 按照EndOfCase来将一个列表分为多个病人列表
            if 'EndOfCase' in bbox_list[i]:
                # 将开的列表添加到一个总的列表中
                pa_divide_list.append(bbox_list[temp:i])
                temp = i + 1
        return pa_divide_list
    def transform(self,single_list):
        '''
        真值 记录不连续，则转为连续
        无记录时则用  下标号，0,0,0,0 填充
        '''
        result=[]
        # 取出所有序号
        all_index=[]
        for i in range(len(single_list)):
            all_index.append(int(single_list[i].split(',')[0]))

        max_index=int(single_list[-1].split(',')[0])
        for i in range(max_index+1):
            # 存在该条记录，则写入
            if i in all_index:
                # 得到all_index中值为i 对应的下标值
                result.append(single_list[all_index.index(i)])

            # 无记录，则填充
            else:
                result.append(str(i)+',0,0,0,0,')
        return result

class get_single_sequence():
    def __init__(self, single_list):
        self.single_list = single_list

    def get_3d(self):
        coordinate_list = []

        # 序列中每个bbox的标识符，0 未使用，1 已使用
        flag = []
        for line in self.single_list:
            coordinate = line.split(',')
            coordinate.pop(0)  # 去掉 序列图像的下标值
            coordinate.pop(-1)  # 去掉 换行符
            coordinate_list.append(np.array(list(map(int, coordinate))))
            flag.append(np.zeros((int(len(coordinate) / 4))))

        # 每行都为list，list里面4个为一组坐标
        coordinate_list_grouping = self.list_2_numpy(coordinate_list)

        node_result = []  # 保存3D坐标的所有结果
        # 遍历所有图像
        for i in range(len(coordinate_list)):
            # 遍历一张图像的所有bbox
            for j in range(int(len(coordinate_list[i]) / 4)):
                # 计算每个bbox与 所有bbox的IOU or 中心点距离

                # 1、先判断flag中 当前bbox 是否已被使： 0 表示未使用  1：表示已使用
                if flag[i][j] == 1:
                    # 已使用，查看下一个框
                    continue
                else:
                    # 未使用,计算 其 与 flag中 未用过的所有bbox的距离

                    # 当前bbox
                    current_bbox = coordinate_list[i][j * 4:j * 4 + 4]
                    # 获取 flag中 未用过的所有bbox
                    coordinate_no_use = self.GetNoUseBbox(coordinate_list_grouping, flag)

                    # 计算 传入bbox 与  flag中未用过的bbox的距离
                    distance = self.Calculated(current_bbox, coordinate_no_use)

                    # 合并bbox
                    merge_result, flag = self.MergeBbox(i, j, distance, opt.distance_threshold, flag)

                    # 将合并结果转化为3D坐标
                    node_3d, info = self.transform(merge_result, coordinate_list_grouping)
                    if 'false' in info:
                        # 一个bbox无法在3D上合并,过滤掉
                        pass
                    else:
                        # 保存淋巴结3D坐标
                        node_result.append(node_3d)
        return node_result

    def list_2_numpy(self, coordinate_list):
        '''
        结果每行都为list，list里面4个为一组坐标
        coordinate_list  list[11,]
        '''
        result_list_2_numpy = []
        for coordinate in coordinate_list:
            temp_list_2_numpy = []
            for i_list_2_numpy in range(0, len(coordinate), 4):  # 步长为4
                temp_list_2_numpy.append(
                    [coordinate[i_list_2_numpy], coordinate[i_list_2_numpy + 1], coordinate[i_list_2_numpy + 2],
                     coordinate[i_list_2_numpy + 3]])
            result_list_2_numpy.append(temp_list_2_numpy)
        return result_list_2_numpy

    def GetNoUseBbox(self, coordinate_list_grouping, flag):
        '''
        从coordinate_list_grouping获取 flag对应位置标记为0的 bbox
        coordinate_list_fixed：list[11,] 每行numpy长度不等
        '''
        # 遍历flag,取出对应位置标记为0的 coordinate_list_grouping的坐标块
        result_GetNoUseBbox = []
        for i_GetNoUseBbox in range(len(flag)):
            temp_GetNoUseBbox = []
            for j_GetNoUseBbox in range(flag[i_GetNoUseBbox].size):
                if flag[i_GetNoUseBbox][j_GetNoUseBbox] == 0:
                    # 未使用
                    temp_GetNoUseBbox.append(coordinate_list_grouping[i_GetNoUseBbox][j_GetNoUseBbox])
                else:
                    # 已使用
                    temp_GetNoUseBbox.append([0, 0, 0, 0])
            result_GetNoUseBbox.append(temp_GetNoUseBbox)
        return result_GetNoUseBbox

    def Calculated(self, current_bbox, coordinate_no_use):
        '''
        计算 传入bbox 与  flag中未用过的bbox  的 欧式距离（or  IOU）
        '''
        # 遍历coordinate_no_use    当一组坐标为[0,0,0,0],表示该坐标已被使用，距离设定为float('inf') 表示正无穷
        result_Calculated = []
        for i_coordinate_no_use in coordinate_no_use:
            temp_Calculated = []
            for j_i_coordinate_no_use in i_coordinate_no_use:
                if sum(j_i_coordinate_no_use) == 0  or  np.sum(current_bbox)==0 :
                    # 标明该坐标已被使用
                    temp_Calculated.append(float('inf'))
                else:
                    # 计算欧式距离
                    distance_Calculated = np.linalg.norm(np.array([((current_bbox[0] + current_bbox[2]) / 2), (
                            (current_bbox[1] + current_bbox[3]) / 2)]) - np.array(
                        [((j_i_coordinate_no_use[0] + j_i_coordinate_no_use[2]) / 2),
                         ((j_i_coordinate_no_use[1] + j_i_coordinate_no_use[3]) / 2)]))
                    temp_Calculated.append(distance_Calculated)
            result_Calculated.append(temp_Calculated)
        return result_Calculated

    def MergeBbox(self, i, j, distance, distance_threshold, flag):
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
        result_MergeBbox = []
        # 刚开始result_MergeBbox全部置为0
        for i_MergeBbox in distance:
            result_MergeBbox.append(np.zeros((len(i_MergeBbox))).tolist())

        # 1、将i,j处置为1
        # 2、从i下一张开始判断，如果每行的最小距离小于等于阈值，则置为1
        result_MergeBbox[i][j] = 1

        # 对应flag位置从0置为1
        if flag[i][j] == 1:
            print('合并出错1')
        else:
            flag[i][j] = 1

        for i_2_MergeBbox in range(i + 1, len(distance)):  # 下标值  从i+1开始
            if min(distance[i_2_MergeBbox]) > distance_threshold:
                break  # 当出现整张不符合时，则 合并结束
            else:
                index = distance[i_2_MergeBbox].index(min(distance[i_2_MergeBbox]))  # 距离<=阈值的横轴下标值
                result_MergeBbox[i_2_MergeBbox][index] = 1

                # 对应flag位置从0置为1
                if flag[i_2_MergeBbox][index] == 1:
                    print('合并出错2')
                else:
                    flag[i_2_MergeBbox][index] = 1

        return result_MergeBbox, flag

    def transform(self, merge_result, coordinate_list_grouping):
        '''
        合并为3D坐标
        '''

        # 一个bbox无法在3D上合并,过滤掉
        count_transform = 0
        for i_transform in merge_result:
            count_transform = count_transform + sum(i_transform)
        if count_transform == 1:
            # 过滤
            return 0, 'false'
        else:
            # 合并为3D坐标

            # 先取出每行中对应的坐标
            temp_transform = []
            temo_index_transform = []
            for i_transform in range(len(merge_result)):  # 序号
                if sum(merge_result[i_transform]) == 0:
                    temp_transform.append([0, 0, 0, 0])
                else:
                    temp_transform.append(coordinate_list_grouping[i_transform][merge_result[i_transform].index(1)])
                    temo_index_transform.append(i_transform)
            # 转为 3D坐标
            z_min, z_max = temo_index_transform[0], temo_index_transform[-1]

            temp_2_transform = np.array(temp_transform[z_min:z_max + 1])
            # x_min,y_min 取最小值，,x_max,y_max取最大值，保证 生成最大范围的bbox
            x_min = np.min(temp_2_transform[:, 0])
            y_min = np.min(temp_2_transform[:, 1])
            x_max = np.max(temp_2_transform[:, 2])
            y_max = np.max(temp_2_transform[:, 3])
            return np.array([x_min, y_min, z_min, x_max, y_max, z_max]), 'true'


