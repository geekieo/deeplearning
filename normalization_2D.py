from normalization import Normalization


class Normalization2D(Normalization):
    '''二维数组归一化'''

    def __init__(self, trainSet):
        # trainSet 为行列数对齐的二维数组
        self.trainSet = trainSet
        self.sampleNum = len(trainSet) #行数，样本数
        self.featNum = len(trainSet[0]) #列数，特征分量数

    ##################### 全局归一化 #####################
    def _2Dto1D(self, X2D):
        '''
        二维数组转换成一维数组
        '''
        X1D = []
        for vec in X2D:
            X1D.extend(vec)
        return X1D

    def _1Dto2D(self,X1D,featNum):
        '''
        将一维数组转换回二维数组
        featNum 为原二维数组每行数据的列数
        '''
        X2D = []; vec = []; count = 0
        for feat in X1D:
            vec.append(feat)
            count +=1
            if count == featNum :
                X2D.append(vec)
                vec = []; count = 0
        return X2D


    def norm2DInGlobal(self, Y2D = None, method="minMaxNorm"):
        '''
        二维数组对全体元素做归一化
        测试样本 Y2D 的列数必须和训练样本列数一致
        适用于特征分量之间起伏不大的数据集，或分量之间非互独的数据集，如图象
        '''
        super().__init__(self._2Dto1D(self.trainSet)) #将二维训练集拉成一维，并送入父类初始化
        if Y2D == None:
            if method == "minMaxNorm":
                out1DN = super().minMaxNorm()
            elif method == "zScoreNorm":
                out1DN = super().zScoreNorm()
            else:
                raise TypeError("method only allowed to be \"minMaxNorm\" or\"zScoreNorm\"")
        else:
            Y1D = self._2Dto1D(Y2D)
            if method == "minMaxNorm":
                out1DN = super().minMaxNorm(Y1D)
            elif method == "zScoreNorm":
                out1DN = super().zScoreNorm(Y1D)
            else:
                raise TypeError("method only allowed to be \"minMaxNorm\" or\"zScoreNorm\"")
        out2DN = self._1Dto2D(out1DN,self.featNum)
        return out2DN

    ################### 按列分量独立归一化 ###################
    def _transpose(self, Y2D):
        '''
        二维数组转置
        '''
        transposed = []
        for i in range(len(Y2D[0])):
            colVec = [vec[i] for vec in Y2D]
            transposed.append(colVec)
        return transposed

    def _norm2DInColumnModel(self):
        '''
        计算测试样本集按列归一化模型
        返回：列归一化模型数组
        '''
        # 如[[1,2,3],[4,5,6],[7,8,9]]
        # [1,4,7],[2,5,8],[3,6,9]分别为独立的归一化样本
        # 需返回三组归一化模型
        normList = []
        for i in range(self.sampleNum):
            colVec = [vec[i] for vec in self.trainSet]
            normList.append(Normalization(colVec))
        return normList

    def norm2DInColumn(self, Y2D=None, method="minMaxNorm"):
        '''
        按列归一化数据
        输入参数：
            测试样本 Y2D：二维数组
                若 Y2D 为空，返回 self.trainSet 归一化结果(默认)，
                否则返回 Y2D 以 self.trainSet 为基准的归一化结果 。
            归一化方法 method：minMaxNorm(默认), zScoreNorm。
        返回：归一化二维数组
        '''
        normList = self._norm2DInColumnModel()  #存放不同分量的归一化模型。
        if method == "minMaxNorm":
            if Y2D == None:
                out2DN = map(lambda n: n.minMaxNorm(), normList)
            else:
                Y2DT = self._transpose(Y2D)  # Y2D 转置,便于按列循环处理
                out2DN = map(lambda n_y: n_y[0].minMaxNorm(n_y[1]),
                           list(zip(normList, Y2DT)))

        elif method == "zScoreNorm":
            if Y2D == None:
                out2DN = map(lambda n: n.zScoreNorm(), normList)
            else:
                Y2DT = self._transpose(Y2D)  # Y2D 转置,便于按列循环处理
                out2DN = map(lambda n_y: n_y[0].zScoreNorm(n_y[1]),
                           list(zip(normList, Y2DT)))
        else:
            raise TypeError("method only allowed to be \"minMaxNorm\" or\"zScoreNorm\"")
        return list(out2DN) # map 转 list


if __name__ == "__main__":
    trainSet = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    norm2D = Normalization2D(trainSet)
    print(norm2D.norm2DInColumn())
    Y2D = [[2, 3, 3], [2, 3, 3], [2, 3, 3], [8, 3, 3]]
    Y2DN = norm2D.norm2DInColumn(Y2D, "zScoreNorm")
    print(norm2D.norm2DInGlobal())