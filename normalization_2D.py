from normalization import Normalization


class Normalization2D(Normalization):
    '''二维数组归一化'''

    def __init__(self, X2D):
        # X2D 为行列数对齐的二维数组
        self.X2D = X2D
        self.rowLen = len(X2D)
        self.colLen = len(X2D[0])
        self.normList = self._normByColumnModel()  #存放不同分量的归一化模型

    def _transpose(self, Y2D):
        '''
        二维数组转置
        '''
        transposed = []
        for i in range(len(Y2D[0])):
            colVec = [vec[i] for vec in Y2D]
            transposed.append(colVec)
        return transposed

    def _normByColumnModel(self):
        '''
        计算测试样本集按列归一化模型
        返回：列归一化模型数组
        '''
        # 如[[1,2,3],[4,5,6],[7,8,9]]
        # [1,4,7],[2,5,8],[3,6,9]分别为独立的归一化样本
        # 返回三组归一化模型
        normList = []
        for i in range(self.colLen):
            colVec = [vec[i] for vec in self.X2D]
            normList.append(Normalization(colVec))
        return normList

    def normByColumn(self, Y2D=None, method="minMaxNorm"):
        '''
        按列归一化数据
        输入参数：
            测试样本 Y2D：二维数组
                若 Y2D 为空，返回 self.X2D 归一化结果(默认)，
                否则返回 Y2D 以 self.X2D 为基准的归一化结果 。
            归一化方法 method：minMaxNorm(默认), zScoreNorm。
        返回：归一化二维数组
        '''
        if method == "minMaxNorm":
            if Y2D == None:
                return map(lambda n: n.minMaxNorm(), self.normList)
            else:
                Y2DT = self._transpose(Y2D)  # Y2D 转置,便于按列循环处理
                return map(lambda n_y: n_y[0].minMaxNorm(n_y[1]),
                           list(zip(self.normList, Y2DT)))

        elif method == "zScoreNorm":
            if Y2D == None:
                return map(lambda n: n.zScoreNorm(), self.normList)
            else:
                Y2DT = self._transpose(Y2D)  # Y2D 转置,便于按列循环处理
                return map(lambda n_y: n_y[0].zScoreNorm(n_y[1]),
                           list(zip(self.normList, Y2DT)))
        else:
            raise TypeError(
                "method of function normByColumn only allowed to be \"minMaxNorm\" or\"zScoreNorm\""
            )


if __name__ == "__main__":
    X2D = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    norm2D = Normalization2D(X2D)
    print(list(norm2D.normByColumn()))
    Y2D = [[2, 3, 3], [2, 3, 3], [2, 3, 3], [8, 3, 3]]
    Y2DN = norm2D.normByColumn(Y2D, "zScoreNorm")
    print(list(Y2DN))