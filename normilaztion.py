from typeError import TypeError

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
        self.correctFormat = self._check()  # 校验 inX 格式
        self.mean = self._mean()
        self.variance = self._variance()
        self.maxValue, self.minValue = self._endValue()

    def _check(self):
        '''格式校验'''
        if type(self.inX).__name__ != 'list' or len(self.inX) == 0:
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
        return sig / len(self.inX)

    def _variance(self):
        '''
        求方差
        '''
        var = 0
        for i in self.inX:
            var += (i - self.mean)**2
        return var

    def linearNorm(self):
        '''
        线性归一化,把一组数值线性映射到[0,1] 区间
        计算公式：inX_norm = (inX - inX_min)/(inX_max-inXmin)
        返回：归一化数组
        '''
        delta = self.maxValue - self.minValue
        return list(map(lambda x: (x - self.minValue) / delta, self.inX))

def test():
    x = [2, 4, 6, 7, 8]
    n = Normalization(x)
    linearNorn = n.linearNorm()
    print(x)
    print(n.mean)
    print(linearNorn)
