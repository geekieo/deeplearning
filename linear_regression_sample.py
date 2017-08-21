from linear_regression import LinearRegssion
from normalization import Normalization

def get_train_dataset():
    '''
    捏造若干条房价数据
    '''
    # # 构造训练数据,注意标签与特征分量一一对应
    # 输入特征向量列表：工作年限
    input_vecs = [[5], [6], [8], [1.4], [10.1]]
    # 输出标签列表，月薪
    labels = [5500, 2300, 7600, 1800, 11400]

    # # 输入特征向量列表：面积
    # input_vecs = [[100], [120], [160], [30], [200]]
    # # 输出标签列表，价格
    # labels = [550,300,760,180,1140]

    ## 输入特征向量列表：地址，面积
    # input_vecs=[[100,100,95],[100,100,130],[100,0,60],[0,100,120],[0,0,120],
    #             [100,0,90],[0,100,80],[0,0,130],[0,0,200],[0,100,130]]
    # # 输出标签列表，价格
    # labels = [500, 650, 40, 600, 300, 70, 360, 280, 500, 700]

    return input_vecs, labels


if __name__ == '__main__':
    '''训练并使用线性回归模型'''
    # 创建感知器，输入参数的特征数为1
    lr = LinearRegssion(1)
    train_vecs, labels = get_train_dataset()
    # 训练样本归一化
    # 注意: 后面的测试样本也得用 train_norm 归一化
    train_norm = Normalization(train_vecs)
    input_vecs = train_norm.minMaxNorm(train_vecs)
    # 训练，迭代10轮，学习速率为0.01
    lr.train(input_vecs, labels, 100, 0.01)
    # 打印训练获得的权重
    print(lr)
    # 测试
    test_vecs = [95]
    test_vecs = train_norm.minMaxNorm(test_vecs)
    print('area = 95, price = %.2f ' % lr.predict(test_vecs))
