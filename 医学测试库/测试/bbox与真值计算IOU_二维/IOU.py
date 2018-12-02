# coding=utf-8
class Judge_Much_IOU():
    # 现在通过计算iou的方法，来切出有瑕疵里面的无瑕疵区域
    @staticmethod
    def calcIOU(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):
        """
        计算IOU
        :param one_x: 预测值bbox 中心坐标 x
        :param one_y: 预测值bbox 中心坐标 y
        :param one_w: 预测值bbox 宽w
        :param one_h: 预测值bbox 高h
        :param two_x: 真值bbox 中心坐标 x
        :param two_y: 真值bbox 中心坐标 y
        :param two_w: 真值bbox 宽w
        :param two_h: 真值bbox 高h
        :return: IOU值
        """
        # 判断是否相交
        if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
            # 计算IOU
            lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
            lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))
            rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
            rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))
            inter_w = abs(rd_x_inter - lu_x_inter)
            inter_h = abs(lu_y_inter - rd_y_inter)
            inter_square = inter_w * inter_h
            union_square = (one_w * one_h) + (two_w * two_h) - inter_square
            calcIOU = inter_square / union_square

        else:
            calcIOU = 0

        return calcIOU

    @staticmethod
    def judge_much_IOU(box_list, this_box_list):
        """
        判断IOU是否匹配
        :param box_list: 预测值一帧的多个bbox eg: box_list[[1,2,3,4],[1,2,3,4]]
        :param this_box_list: 真值bbox eg: this_box_list [xmin,ymin,xmax,ymax]
        :return: 匹配返回1，不匹配返回0
        """
        xmin, ymin, xmax, ymax = this_box_list
        # 计算真值bbox中心及宽高
        two_x, two_y, two_w, two_h = xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax - xmin, ymax - ymin
        for onelist in box_list:
            xmin_, ymin_, xmax_, ymax_ = onelist
            xmin_, ymin_, xmax_, ymax_ = int(xmin_), int(ymin_), int(xmax_), int(ymax_)
            # 计算预测值bbox中心及宽高
            one_x, one_y, one_w, one_h = xmin_ + (xmax_ - xmin_) / 2, ymin_ + (
                    ymax_ - ymin_) / 2, xmax_ - xmin_, ymax_ - ymin_
            # 计算IOU
            result = Judge_Much_IOU.calcIOU(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h)

            if result > 0.9:
                # 这里设置阈值
                return 1
        return 0  # 表示该帧没有与真值相交的bbox


judge_much_IOU = Judge_Much_IOU()
