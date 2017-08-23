from perceptron.perceptron import Perceptron
from activator import linear

class LinearRegssion(Perceptron):
    def __init__(self, input_num):
        '''
        初始化线性回归模型，设置输入参数的个数
        '''
        # 感知机取消阶跃激活函数即为线性回归
        Perceptron.__init__(self, input_num, linear)