class filter:
    '''
    过滤候选3D块
    '''
    def __init__(self,predict_3D):
        self.predict_3D = predict_3D
    def get_result(self):
        # 遍历所有序列
        result=[]
        for single_group in self.predict_3D:
            # 遍历序列内所有候选3D块
            single_group_result=[]
            for i_single_group in  single_group:
                z_min=i_single_group[2]
                z_max=i_single_group[5]
                if z_max-z_min >2: # 至少三层
                    single_group_result.append(i_single_group)
            result.append(single_group_result)
        return result



