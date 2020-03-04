import numpy as np
import scipy.linalg as la
from SimpleNeuralNet import NeuralNet
from sklearn.datasets import load_iris

def run(n,d):
    y = np.ones((n,1))
    y[int(n/2):] = 2
    y = y.astype(int)
    X = np.random.random((n,d))
    idx_row, idx_col = np.where(y==1)
    X[idx_row,0] = 0.1+X[idx_row,0]
    idx_row, idx_col = np.where(y==-1)
    X[idx_row,0] = -0.1-X[idx_row,0]
    U = la.orth(np.random.random((d,d)))
    X = np.dot(X,U)
    return (X,y)

def test():
    np.random.seed(0)
    n = 1280
    f = 5
    cl = 2
    data,labels = run(n,f)

    # data, labels = load_iris(return_X_y=True)
    # labels = labels + np.ones_like(labels)
    labels = np.array([x[0] for x in labels])
    n = len(data)
    f = len(data[0])
    # print(labels)
    model = NeuralNet(f,cl,lrate=0.1,verbose=True)
    model.fit(data,labels,max_iter=200)
    result = [model.predict(x) for x in data]
    labels = labels.reshape((-1,))
    print(labels,result)
    accuracy = np.where(result==labels,1,0).sum()
    accuracy /= len(result)
    # print(result)
    print(accuracy)
    
    # print(data[0],forward_result)
    # print('total prob',np.sum(forward_result))

if __name__=='__main__':
    test()