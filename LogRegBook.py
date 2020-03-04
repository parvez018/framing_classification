import numpy as np

class LogRegression():
    def __init__(self,in_feature,out_class,lr = 0.1):
        self.theta = np.zeros((out_class*(in_feature+1),))
        self.in_features = in_feature
        self.out_class = out_class
        self.learning_rate = lr
    
    def getf(self,x,y):
        if y<1 or y>self.out_class:
            raise Exception('invalid output label')
        print(x,y)
        f = np.zeros(np.shape(self.theta))
        f[(y-1)*(self.in_features+1)] = 1
        f[(y-1)*(self.in_features+1)+1:(y)*(self.in_features+1)+1] = x
        return f

    def predict_prob(self,X):
    
    def toLabel(self,z):
        return np.argmax(z,axis=1)

    def fit(self,X,y,maxiter=100):
        n = len(X)
        epsilon = 10^(-6)
        loss_train = []
        iters = 0
        one_hot = np.zeros((n,self.out_class))
        # print('one_hot',np.shape(one_hot))
        for i in range(n):
            one_hot[i][y[i]-1]=1
        # print(one_hot)
        while iters<maxiter:
            iters += 1
            probs = self.predict_prob(X)
            smax = one_hot-probs
            delta_w = -X.T.dot(smax)/n
            # gradient = np.sum(gradient,axis=0)
            self.w -= self.learning_rate*delta_w
            delta_b = -(np.sum(smax,axis=0)/n).reshape(-1,1).T/n
            self.b -= self.learning_rate*(delta_b.T)
            
            # print(smax)
            loss = np.where(smax>0,smax,0)
            loss = np.sum(loss,axis=1)
            loss = np.mean(loss)
            print(loss)
        print('loss',loss)

