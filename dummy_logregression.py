import numpy as np
import scipy.linalg as la
from SimpleLogRegression import LogRegression
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
    f = 1000
    cl = 3
    data,labels = run(n,f)
    data, labels = load_iris(return_X_y=True)
    labels = labels + np.ones_like(labels)
    # data = [[1]*f]
    # data.append([22]*f)
    # data.append([200]*f)
    # labels = [0,1,2]
    # print(data,labels)
    # data = np.array([[-1,-1],[1,1],[-1,1],[1,-1]])
    # labels = np.array([2,1,1,1])
    n = len(data)
    f = len(data[0])
    # print(labels)
    model = LogRegression(f,cl,lrate=0.1,verbose=True)
    model.fit(data,labels,max_iter=500)
    result = [model.predict(x) for x in data]
    labels = labels.reshape((-1,))
    
    accuracy = np.where(result==labels,1,0).sum()
    accuracy /= len(result)
    print(labels,result)
    print(accuracy)

def cv_test():
    pass
if __name__=='__main__':
    # test()