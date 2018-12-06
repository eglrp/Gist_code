'''
xml文件名序号从1开始，注意改为从0开始
'''
import glob as gb
import xml.etree.ElementTree as ET
import numpy as np
def parse_rec(index,filename):
    """
    解析一个 PASCAL VOC xml file
    """
    tree = ET.parse(filename)
    # 存储一张图片中的所有物体
    objects = []
    objects.append(index)
    # 遍历一张图中的所有物体
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        # 从原图左上角开始为原点，向右为x轴，向下为y轴。左上角（xmin，ymin）和右下角(xmax,ymax)
        objects.append(int(float(bbox.find('xmin').text)))
        objects.append(int(float(bbox.find('ymin').text)))
        objects.append(int(float(bbox.find('xmax').text)))
        objects.append(int(float(bbox.find('ymax').text)))

    return objects



def get_bbox_coordinate(i_single_xml_path):
    '''
    得到xml里所有bbox坐标
    '''
    index = int(i_single_xml_path.replace('.xml', '')[-5:]) - 1  # xml文件名序号从1开始，故-1 变为从0开始
    single_bbox_info = parse_rec(index,i_single_xml_path)
    return single_bbox_info


def get_single_bbox(single_xml_path):
    '''
    通过 一个序列的地址 得到 该序列的bbox
    '''
    single_xml_path = gb.glob(single_xml_path + "/*.xml")
    single_xml_path.sort()  # 排序

    # 遍历所有xml文件
    all_bbox_info=[]
    for i_single_xml_path in single_xml_path:
        all_bbox_info.append(get_bbox_coordinate(i_single_xml_path))
    return all_bbox_info
def write_txt(bbox_txt_dir,result):
    '''
    将结果写入txt文件中
    '''
    # 将坐标写入txt
    txt_file = open(bbox_txt_dir, 'w')
    txt_file.write('index_sequence，x_min,y_min,x_max,y_max\n')
    # 遍历每个序列
    for single_result in result:
        for info in single_result:
            for single_info in info:
                txt_file.write(str(single_info) + ',')
            txt_file.write('\n')
        txt_file.write('==============EndOfCase==============')  # 一个序列结束
        txt_file.write('\n')
    txt_file.write('\n')

def get_bbox(xml_path):
    bbox_list=[]
    # 遍历每个序列
    for single_xml_path in  xml_path:
        bbox_list.append(get_single_bbox(single_xml_path))
    return bbox_list

def test():
    xml_dir = '/home/bobo/data/test/bbox_xml/'
    bbox_txt_dir = 'truth_bounding_box.txt'
    xml_path = gb.glob(xml_dir + "*")
    xml_path.sort()  # 排序

    result = get_bbox(xml_path)
    # 写入txt
    write_txt(bbox_txt_dir,result)




if __name__ == '__main__':
    test()
