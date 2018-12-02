import csv
import numpy as np
from config import opt
class getTruthCoordinate():
    '''
    得到所有 真值3D块坐标
    3D块坐标 标准（x_min,y_min,z_min,x_max,y_max,z_max）
    '''
    def __init__(self):
        self.node_mark_dir = opt.node_mark_dir

    def get_3d_coordinate(self):
        csv_reader = self.load_csv(self.node_mark_dir)
        return self.save_gruop(csv_reader)

    def load_csv(self,csv_dir):
        """
        加载csv数据
        """
        csvFile = open(csv_dir, "r")
        return csv.reader(csvFile)

    def save_gruop(self,csv_reader):

        node_mark_list = []  # 保存所有淋巴结标注信息
        for item in csv_reader:
            node_mark_list.append(item)

        node_mark_list.pop(0) # 去掉标题
        # 按照PA分块保存
        list_sorted_PA = []
        for i in range(len(node_mark_list)):
            list_sorted_PA.append(node_mark_list[i][0])

        list_sorted_PA = list(set(list_sorted_PA))
        list_sorted_PA.sort()

        list_Grouping = []
        # 分块保存
        for j in range(len(list_sorted_PA)):
            temp = []
            for k in range(len(node_mark_list)):
                if list_sorted_PA[j] == node_mark_list[k][0]:
                    temp.append(node_mark_list[k])
            list_Grouping.append(temp)

        # 此时list_Grouping为分组存储

        mark_coordinate_all = []
        # 遍历每一个病例
        for i in range(len(list_Grouping)):
            mark = list_Grouping[i]
            mark_coordinate = []
            # 遍历每一个淋巴结
            for ii in range(len(mark)):
                x, y, longest_in_xy, begin, end = int(mark[ii][1]), int(mark[ii][2]), float(mark[ii][3]), int(
                    mark[ii][4]), int(mark[ii][5])
                x_min, x_max = int(x - longest_in_xy / 2) - 1, int(x + longest_in_xy / 2) + 1
                y_min, y_max = int(y - longest_in_xy / 2) - 1, int(y + longest_in_xy / 2) + 1
                z_min, z_max = begin, end

                mark_coordinate.append(np.array([x_min, y_min, z_min, x_max, y_max, z_max]))
            mark_coordinate_all.append(mark_coordinate)

        # mark_coordinate_all为 分组存储的3D块的坐标
        return mark_coordinate_all