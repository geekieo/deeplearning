from normalization import Normalization

class Normalization2D(Normalization):
    '''二维数组归一化'''

    def __init__(self, X2D):
        # X2D 为行列数对齐的二维数组
        self.X2D = X2D
        self.rowLen = len(X2D)
        self.colLen = len(X2D[0])
        self.normList = self._normByColumnModel() #存放不同分量的归一化模型


    def _normByColumnModel(self):
        '''
        计算按列归一化模型
        '''
        # 如[[1,2,3],[4,5,6],[7,8,9]]
        # [1,4,7],[2,5,8],[3,6,9]分别为独立的归一化样本
        # 返回三组归一化模型
        normList = []
        for i in range(self.colLen):
            colVec = [vec[i] for vec in self.X2D]
            normList.append(Normalization(colVec))
        return normList
    
    def normByColumn(self, Y2D = None, method = "minMaxNorm"):
        '''
        按列归一化数据
        输入参数：若 Y2D 为空，返回 self.X2D 归一化结果(默认)，
            否则返回 Y2D 以 self.X2D 为基准的归一化结果 
        归一化方法：minMaxNorm(默认),zScoreNorm
        '''
        if method == "minMaxNorm":
            if Y2D == None:
                return map(lambda n: n.minMaxNorm(),self.normList)

        elif method == "zScoreNorm":
            if Y2D == None:
                return map(lambda n: n.zScoreNorm(),self.normList)

        else:
            raise TypeError("method only allowed to be \"minMaxNorm\" or\"zScoreNorm\"")
        
        # for i in range(len(vec2D)):
        #     colVec = [vec[i] for vec in vec2D]


if __name__ == "__main__":
    X2D = [[1,2,3],[4,5,6],[7,8,9]]
    norm2D = Normalization2D(X2D)
    # normList = norm2D._normByColumnModel()
    # print(normList)
    print(list(norm2D.normByColumn()))