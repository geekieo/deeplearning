from perceptron import Perceptron
from activator import step


def get_training_dataset():
    '''
    基于 and 真值表构建训练数据
    '''
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，需要与输入向量一一对应
    # [1,1]->1, [0,1]->0, [1,0]->0, [0,0]->0
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    '''
    使用and真值表数据训练感知器
    '''
    perceptron = Perceptron(2, step)
    # 训练，迭代10轮，学习速率为 0.1
    input_vecs, labels = get_training_dataset()
    perceptron.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return perceptron


if __name__ == '__main__':
    # 训练 and 感知器
    and_perceptron = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perceptron)
    # 测试
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]),
          '1 and 0 = %d' % and_perceptron.predict([1, 0]),
          '0 and 1 = %d' % and_perceptron.predict([0, 1]),
          '0 and 0 = %d' % and_perceptron.predict([0, 0]),
          sep='\n')
