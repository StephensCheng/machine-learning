import matplotlib.pyplot as plt
import random
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, model_selection
from sklearn import svm
import numpy as np

# step1  ppt中要用到的一些例图
# def step_function(x):
#     return np.array(x > 0, dtype=np.int)
#
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# x = np.arange(-2.0, 2.0, 0.1)
# y = step_function(x)
# y1=sigmoid(x)
#
# plt.plot(x, y,label="step-function")
# plt.plot(x,y1,label="sigmoid")
# plt.legend(loc="best")
# plt.ylim(-0.1, 1.1)
# plt.show()

# step2 准备数据
s = True
if s:
    bc = datasets.load_breast_cancer()
    X = bc.data[:150,:2]
    Y = bc.target[:150]

    x_train, x_test = [], []
    y_train, y_test = [], []
    for i in range(100):
        if i < 110:
            x_train.append([1.0, float(X[i][0]), float(X[i][1])])
            y_train.append(int(Y[i]))
        else:
            x_test.append([1.0, float(X[i][0]), float(X[i][1])])
            y_test.append(int(Y[i]))

    np.save("./Data/LR/train_data.npy", np.array(x_train))
    np.save("./Data/LR/train_target.npy", np.array(y_train))
    np.save("./Data/LR/test_data.npy", np.array(x_test))
    np.save("./Data/LR/test_target.npy", np.array(y_test))
else:
    x_train = np.load("./Data/LR/train_data.npy")
    y_train = np.load("./Data/LR/train_target.npy")
    x_test = np.load("./Data/LR/test_data.npy")
    y_test = np.load("./Data/LR/test_target.npy")


#
# px1,px2=[],[]
# for i in range(40):
#     if y_test[i]==1:
#         px1.append(x_test[i])
#     else:
#         px2.append(x_test[i])
# px1=np.array(px1)
# px2=np.array(px2)
# plt.scatter(px1[:,0],px1[:,1],c="red")
# plt.scatter(px2[:,0],px2[:,1],c="blue")
# plt.show()


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def gradascent(dataMat, labelMat):
    dataMatrix = np.mat(dataMat)
    classLabels = np.mat(labelMat).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (h-classLabels)
        weights = weights - alpha * dataMatrix.transpose() * error
    return weights

def stocGraAscent(dataMatrix,matLabel):
    m,n=np.shape(dataMatrix)
    matMatrix=np.mat(dataMatrix)

    w=np.ones((n,1))
    alpha=0.001
    num=20  #这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    for i in range(num):
        for j in range(m):
            error=sigmoid(matMatrix[j]*w)-matLabel[j]
            w=w-alpha*matMatrix[j].transpose()*error
    return w


def bestfit(weights):
    px1, px2 = [], []
    for i in range(100):
        if y_train[i] == 1:
            px1.append(x_train[i])
        else:
            px2.append(x_train[i])
    px1 = np.array(px1)
    px2 = np.array(px2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(px1[:, 1], px1[:, 2], c="red")
    ax.scatter(px2[:, 1], px2[:, 2], c="blue")
    x = np.arange(8.0, 25.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.show()


def main():
    weights = stocGraAscent(x_train, y_train).getA()  # getA()转化为数组
    print(weights)
    bestfit(weights)


if __name__ == "__main__":
    main()
