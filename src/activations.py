'''
激活函数
'''


def sigmoid(x):
    '''
    sigmoid 激活函数
    设 y = sigmoid(x)
    则 dy/dx = y(1-y)
    '''
    from math import exp
    return 1 / (1 + exp(-x))

def step(x):
    '''
    0,1 阶跃激活函数
    '''
    return 1 if x > 0 else 0

def linear(x):
    '''
    线性激活
    啥也没干
    '''
    return x

def tanh(x):
    '''
    双曲正切激活函数
    设 y = tanh(x)
    则 dy/dx = (1+y)(1-y) = 1-y^2
    '''
    from math import tanh
    return tanh(x)

def relu(x):
    return 0 if x<0 else x

def elu(x,alpha):
    from math import exp
    return x if x>0 else alpha*(exp(x)-1)

def selu(x):
    _scale = 1.0507009873554804934193349852946
    _alpha = 1.6732632423543772848170429916717
    return _scale*elu(x,_alpha)

def _plot():
    pass