class Perceptron(object):
    # 注释中“打包在一起”的意思是，使打包对象成为同一次运算的处理对象

    def __init__(self, input_num, activator):
        '''
        初始化感知机，设置输入参数个数，以及激活函数。
        参数个数为样本特征分量个数，将为之初始化权重系数
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
                list(map(lambda x_w: x_w[0]*x_w[1],
                    list(zip(input_vec, self.weights))))
            ) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组特征向量，与每个向量对应的 label；训练轮数；学习率
        '''
        # 把输入和输出打包在一起，生成样本列表 [(input_vec, label),...]
        # 即把一个样本的特征和其标签打包，每个训练样本是 (input_vec, label)
        samples =  list(zip(input_vecs, labels))
        for _ in range(iteration):
            self._one_iteration(samples, rate)

    def _one_iteration(self, samples, rate):
        '''
        一次迭代，把所有训练样本过一遍，
        每个训练样本将更新一遍权重系数
        '''

        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        权重更新公式为误差驱动 error-driven，
        和逻辑回归权重更新公式一模一样
        '''
        # 把 input_vec[x1,x2,x3,...] 和 weights[w1,w2,w3,...] 打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 权重更新规则：wi = wi + rate * error * xi ; bias 同理
        error = label - output
        self.weights = list(map(
            lambda w_x: w_x[0] + rate * error * w_x[1],
            list(zip(self.weights, input_vec))
        ))
        # 更新 bias
        self.bias += rate * error