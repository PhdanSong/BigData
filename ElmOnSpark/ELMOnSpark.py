import numpy as np
import os

def compute(path,rows,cols):

    '''
    :param path: 要读取的文件路径
    :param rows: 生成Array的行数
    :param cols: 生成Array的列数
    :return: rows行cols列的Array类型的对象
    '''

    myArray = np.zeros((rows, cols), dtype=float)
    f = open(path)
    lines = f.readlines()
    myArray_row = 0
    for line in lines:
        list = line.strip('\n').split(' ')
        myArray[myArray_row:] = list[0:cols]
        myArray_row += 1
    return myArray

def eachFile(filepath):

    '''
    :param filepath:待读取的文件夹
    :return:
        list: 文件名列表
        count：文件数量
    '''

    count = 0
    pathDir = os.listdir(filepath)
    list = []
    for allDir in pathDir:
        list.append(allDir)
        count += 1
    return list,count


def readFile(filename):

    """
    :param filename: 待读取的文件名
    :return: 包含该文件下的所有内容的Array对象
    """

    fopen = open(filename, 'r')
    lines = []
    for row in fopen.readlines():
        lines.append(row.split())
    return lines

def sigmoid(z):

    """
    :param z: 待处理数据
    :return: sigmoid
    """

    s = 1 / (1 + np.exp(-z))
    return s

def readDir(filePathC):

    """
    :param filePathC: 待读取文件夹的路径
    :return: 存放该文件夹下所有文件内容的array对象
    """
    fnames_list, fileNum = eachFile(filePathC)
    v = []
    for fname in fnames_list:
        lines = readFile(filePathC + "\\" + fname)
        v.extend(lines)
    V = np.array(v).astype('float32')
    return V


if __name__ == '__main__':

    #   用于测elm精度的数据集规模
    numData = 1000
    #   隐层节点个数
    numHiddenLayerNode = 100
    filePathV = "E:\\ELMOnSpark\\" + str(numHiddenLayerNode) + "\\V"
    filePathU = "E:\\ELMOnSpark\\" + str(numHiddenLayerNode) + "\\U"
    #   导入W权重矩阵
    W = compute("E:\\ELMOnSpark\\" + str(numHiddenLayerNode) + "\\W\\part-00000", 4, numHiddenLayerNode)
    #   导入U权重矩阵
    U = readDir(filePathU)
    #   导入V权重矩阵
    V = readDir(filePathV)

    lamda = 0
    lammda_plus = np.eye(numHiddenLayerNode,numHiddenLayerNode)*lamda
    U_I = np.mat(U+lammda_plus).I
    #   计算beta矩阵
    beta = np.dot(U_I,V)
    #   保存生成的beta矩阵
    np.savetxt("E:\\ELMOnSpark\\beta.txt", beta)

    #   读入测试数据集
    testArrayOriginal=compute("E:\\ELMOnSpark\\test100.txt", numData, 4)
    testArray=compute("E:\\ELMOnSpark\\test100.txt", numData, 4)
    #   处理测试数据集
    testArray[:,-1] = 1

    #   计算测试数据集的H值
    H = sigmoid(np.dot(testArray, W))
    #   测试数据集的实际输出
    y_pred = np.dot(H, beta)

    y_pred_label = np.argmax(y_pred, axis = 1) + 1
    y_original_label = testArrayOriginal[:,-1].reshape([-1, 1])
    #   测试数据集的精度计算
    acc = np.sum(y_pred_label == y_original_label) / y_pred_label.shape[0]
    print("accuracy is %f" %acc)

