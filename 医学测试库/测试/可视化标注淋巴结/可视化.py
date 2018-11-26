import csv
import glob as gb
# 读取csv至字典
csvFile = open("node_mark.csv", "r")
reader = csv.reader(csvFile)


node_mark_list = []  # 保存所有淋巴结标注信息
for item in reader:
    node_mark_list.append(item)

# 按照PA分块保存
list_sorted_PA=[]
for i in range(len(node_mark_list)):
    list_sorted_PA.append(node_mark_list[i][0])

list_sorted_PA=list(set(list_sorted_PA))
list_sorted_PA.sort()

list_Grouping=[]
# 分块保存
for j in range(len(list_sorted_PA)):
    temp = []
    for k in range(len(node_mark_list)):
        if list_sorted_PA[j] ==  node_mark_list[k][0]:
            temp.append(node_mark_list[k])
    list_Grouping.append(temp)

# 此时list_Grouping为分组存储



# 加载dcm
dataset_path='/home/bobo/data/NeckLymphNodes/dataset/'  # 数据集地址
dcm_path_dicom = gb.glob(dataset_path + "dcm/*")
dcm_path_dicom.sort()  # 排序


# list_Grouping为分组存储淋巴结信息     dcm_path_dicom为分组存储dcm数据
# 开始可视化

mark=list_Grouping[0]
dcm=gb.glob(dcm_path_dicom[0] + "/*.dcm")
dcm.sort()

print()



