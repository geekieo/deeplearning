from typeError import TypeError
from math import sqrt

class Normalization():
    '''
    数据归一化
    '''


    def __init__(self, inX):
        '''
        初始化数据，输入 list 数组
        '''
        # inX 为待处理数组
        self.inX = inX
        self.length = len(inX)
        self.correctFormat = self._check()  # 校验 inX 格式
        self.mean = self._mean()
        self.variance = self._variance()
        self.stdDeviation = self._stdDeviation()
        self.maxValue, self.minValue = self._endValue()

    def _check(self):
        '''格式校验'''
        if type(self.inX).__name__ != 'list' or self.length == 0:
            raise TypeError("Please check the value, make sure it is a list!")

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

    def minMaxNorm(self):
        '''
        最大最小值归一化
        把一组数值线性映射到 [0,1] 区间
        计算公式：x_norm = (x - inX_min) / (inX_max - inX_min)
        返回：归一化的数组
        适用环境：不需要考虑特征分量之间的量纲差异，适用于大部分情况
        '''
        delta = self.maxValue - self.minValue
        return list(map(lambda x: (x - self.minValue) / delta, self.inX))
    
    
    def zScoreNorm(self):
        '''
        0均值标准差归一化
        把一组数组映射到均值为0的标准差归一化区间
        计算公式：x_norm = (x - inX_mean) / stdDeviation
        返回：归一化数组
        适用环境：需要消除特征分量之间量纲差异
        '''
        return list(map(lambda x:(x-self.mean)/self.stdDeviation, self.inX))



def test():
    x = [2, 4, 6, 7, 8]
    n = Normalization(x)
    minmaxnorm = n.minMaxNorm()
    zscorenorm = n.zScoreNorm()
    print(x)
    print(n.mean, n.stdDeviation)
    print(minmaxnorm)
    print(zscorenorm)