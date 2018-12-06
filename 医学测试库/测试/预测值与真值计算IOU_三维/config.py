class DefaultConfig():

    bounding_box_dir='truth_bounding_box.txt' # 候选bbox坐标（带标题）
    truth_bounding_box_dir='truth_bounding_box.txt' # 真实标记的淋巴结坐标（带标题）

    distance_threshold = 11  # 将中心点距离为5mm以内的bbox在三维上合并

    threshold_3d_iou = 0.8 # 当三维IOU超过阈值时，认为 三维上框出真实淋巴结


#初始化该类的一个对象
opt = DefaultConfig()