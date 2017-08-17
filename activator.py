def sigmoid(x):
    '''
    sigmoid 激活函数
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