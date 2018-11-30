'''
将json转化为mask文件   在对应环境的命令行执行py文件
'''


import os
import glob as gb
import numpy as np
# file_dir='/home/bobo/data/CT_Nodes/MED_ABD_LYMPH_ANNOTATIONS/'
file_dir='G:\\aaaa\\'
img_path_dicom = gb.glob(file_dir+"*\\*.json") #查找文件名末尾叫ces.txt
img_path_dicom.sort()  #排序
for i in range(len(img_path_dicom)):
    os.system("labelme_json_to_dataset  "+ img_path_dicom[i])

