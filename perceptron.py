def sigmoid(x):
    import math
    return 1/(1+math.exp(-x))

class Perceptron(object):

    def __init__(self, input_num, activator):
        '''
        初始化感知机，设置输入参数个数，以及激活函数。
        激活函数的类型为 double -> double
        '''
        # 激活函数
        self.activator = activator
        # 权重向量初始化0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重，偏置项
        print(instance) 将调用本方法打印返回值
        '''
        return 'weights\t: %s\nbias\t: %f\n' %(self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入样本向量，输出感知器的计算结果
        '''
        # activator(加权和 + bias)
        # 把 input_vec[x1,x2,x3...] 和 weights[w1,w2,w3...]打包在一起
        # 利用 zip 生成 [(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用 map 函数计算[x1*w1,x2*w2,x3*w3]
        # 最后利用 reduce 求和
        from functools import reduce
        return self.activator(
            reduce(lambda a, b: a+b,
                map(lambda xw: xw[0]*xw[1],
                    list(zip(input_vec, self.weights)))
            ) + self.bias)
