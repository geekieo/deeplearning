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
        inList = self.inX if inY == None else inY
        if type(inList).__name__ != 'list':
            raise TypeError(
                "Please check the argument, make sure the type is list!")
        length = len(inList)
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


if __name__ == '__main__':
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