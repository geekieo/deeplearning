"""
全链接神经网络(BP神经网络)
结构：
       Network
       /      \
    Layer   Connections
      |         |
    Node —— Connection    
Network 神经网络对象，提供API接口。它由若干层对象组成以及连接对象组成。
Layer 层对象，由多个节点组成。
Node 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
Connection 每个连接对象都要记录该连接的权重。
Connections 仅仅作为Connection的集合对象，提供一些集合操作。

变量及术语说明：
时间凝固，观察某一层，涉及方向的结构变量：
    前层，f_stream
    后层，b_stream
时间流动，观察某一层，涉及方向的动作术语：
    前馈，从当前层到后一层
    反向，从当前层到前一层
"""
from activator import sigmoid
from functools import reduce


class Node(object):
    '''
    节点类，负责记录和维护节点自身信息，
    以及与这个节点相关的前后层连接，
    实现输出值 output 和误差项 error 的计算。
    '''

    def __init__(self, layer_index, node_index):
        '''
        构造节点对象
        layer_index: 节点所属层的编号
        node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.f_stream = [] #前层节点
        self.b_stream = [] #后层节点
        self.output = 0
        self.error = 0

    def set_output(self, output):
        '''
        设置节点输出值
        如果节点为输入层，需调用该函数
        '''
        self.output = output

    def append_f_stream_connection(self, conn):
        '''
        添加一个到前一层节点的链接
        '''
        self.f_stream.append(conn)

    def append_b_stream_connection(self, conn):
        '''
        添加一个到后一层节点的链接
        '''
        self.b_stream.append(conn)

    def calc_output(self):
        '''
        计算各层前馈结果
        前馈传播公式：output = sigmoid（W·X)
            其中， W 为 weights 前馈权重，行向量；X 为节点输入，来自前一层输出，列向量，
            W·X 为输入值的加权和，sigmoid() 为激活函数
        '''
        # ret 为 reduce 时的加权项缓存
        output = reduce(
            lambda ret, conn: ret + conn.f_stream_node.output * conn.weight,
            self.f_stream)

    def calc_output_layer_error(self, label):
        '''
        计算输出层节点误差梯度
        输出层BP公式：error = output * (1-output)*(label - output)
        '''
        self.error = self.output * (1 - self.output) * (label - self.output)

    def calc_hidden_layer_error(self):
        '''
        计算隐层节点误差梯度
        隐层BP公式：error = output*(1-output)*Σ(w·out_error)
            error 为当前节点方向传播误差值；output 为当前节点前馈输出值；
            W 为当前节点前馈权重，有多个权重；out_error 为当前节点前馈节点误差，对应权重，有多个误差；
            当前节点的误差为前馈层的节点误差加权和，不同节点的前馈节点误差相同，但不同节点的前馈权重不同。
        '''
        # ret 为 reduce 时的加权项缓存
        b_stream_error = reduce(
            lambda ret, conn: ret + conn.b_stream_node.error * conn.weight,
            self.b_stream, 0.0)
        self.error = self.output *(1-self.output)*b_stream_error

    def __str__(self):
        '''
        重写对象打印方法，打印节点的信息
        '''
        node_str = 'L%u-N%u: output: %f error: %f' % (self.layer_index,
                                                      self.node_index,
                                                      self.output, self.error)
        b_stream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                              self.b_stream, '')
        f_stream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                              self.f_stream, '')
        return node_str + '\n\tb_stream:' + b_stream_str + '\n\tf_stream:' + f_stream_str


class ConstNode(object):
    '''
    常数项节点
    实现一个输出恒为1的节点，计算偏执项 wb 时需要
    1 的地位等同于其他节点输入的特征变量 x  
    公式 bias_output = 1*wb
    '''

    def __init__(self, layer_index, node_index):
        '''
        构造节点对象
        layer_index: 节点所属层的编号
        node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.b_stream = [] #后层节点
        self.output = 1
        self.error = 0

    def append_b_stream_connection(self, conn):
        '''
        添加一个到下一层节点的链接
        '''
        self.b_stream.append(conn)

    def calc_hidden_layer_error(self):
        '''
        计算隐层节点误差梯
        隐层BP公式：error = output*(1-output)*Σ(w·out_error)
        '''
        b_stream_error = reduce(
            lambda ret, conn: ret + conn.b_stream_node.error * conn.weight,
            self.b_stream, 0.0)
        self.error = self.output * (1 - self.output) * b_stream_error
   
    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        b_stream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.b_stream, '')
        return node_str + '\n\tb_stream:' + b_stream_str