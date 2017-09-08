from typeError import TypeError
from math import sqrt


class Normalization():
    '''
    一维数组归一化
    '''

    def __init__(self, inX):
        '''
        初始化数据，输入一维数组
        '''
        # inX 为待处理数组
        self.inX = inX
        self.length = self._check()  # 校验 inX 格式，正确则返回其长度
        self.mean = self._mean()
        self.variance = self._variance()
        self.stdDeviation = self._stdDeviation()
        self.maxValue, self.minValue = self._endValue()

    def _check(self, inY=None):
        '''格式校验'''
        vec = self.inX if inY == None else inY
        if type(vec).__name__ != 'list':
            raise TypeError(
                "Please check the argument, make sure the type is list!")
        length = len(vec)
        if length == 0:
            raise TypeError("Please check the argument, it can't be empty!")
        return length

    def _endValue(self):
        '''
        求最大最小值
        '''
        maxValue = minValue = self.inX[0]
        for x in self.inX:
            if x > maxValue:
                maxValue = x
            if x < minValue:
                minValue = x
        return maxValue, minValue

    def _mean(self):
        '''求均值'''
        sig = 0
        for i in self.inX:
            sig += i
        return sig / self.length


    def _variance(self):
        '''
        求方差
        '''
        var = 0
        for i in self.inX:
            var += (i - self.mean)**2
        return var / self.length


    def _stdDeviation(self):
        '''
        求标注差
        '''
        return sqrt(self.variance)


    def minMaxNorm(self, inY=None):
        '''
        最大最小值归一化
        把一组数值线性映射到测试样本的所映射的 [0,1] 区间
        参数：if inY == None 返回样本归一化数据 else 返回 inY 在 inX 归一化参数下的归一化数据
        返回：归一化的数组
        计算公式：x_norm = (x - inX_min) / (inX_max - inX_min)
        适用环境：不需要考虑特征分量之间的量纲差异，适用于大部分情况
        '''
        delta = self.maxValue - self.minValue
        return list(
            map(lambda x: (x - self.minValue) / delta, self.inX
                if inY == None else inY))


    def zScoreNorm(self, inY=None):
        '''
        0均值标准差归一化
        把一组数组映射到均值为0的标准差归一化区间
        参数：if inY == None 返回样本归一化数据 else 返回 inY 在 inX 归一化参数下的归一化数据
        返回：归一化数组
        计算公式：x_norm = (x - inX_mean) / stdDeviation
        适用环境：需要消除特征分量之间量纲差异
        '''
        return list(
            map(lambda x: (x - self.mean) / self.stdDeviation, self.inX
                if inY == None else inY))

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

if __name__ == '__main__':
    # 1维归一化测试
    inX = [2, 4, 6, 7, 8]
    inY = [56, 48, 59]
    n = Normalization(inX)
    print(inX)
    print(inY)
    print(n.mean, n.stdDeviation, sep='; ')
    print(n.minMaxNorm())
    print(n.minMaxNorm(inY))
    print(n.zScoreNorm())
    print(n.zScoreNorm(inY))

    # 2维归一化测试
    trainSet = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    norm2D = Normalization2D(trainSet)
    print(norm2D.norm2DInColumn())
    Y2D = [[2, 3, 3], [2, 3, 3], [2, 3, 3], [8, 3, 3]]
    Y2DN = norm2D.norm2DInColumn(Y2D, "zScoreNorm")
    print(norm2D.norm2DInGlobal())
    