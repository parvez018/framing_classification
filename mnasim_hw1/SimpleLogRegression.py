import numpy as np

class LogRegression():
    def __init__(self,n_feature,n_class,lrate = 0.1,verbose=False):
        self.verbose = verbose
        self.n_input = n_feature
        self.n_class = n_class
        self.lrate = lrate
        self.model = {}

        # self.model['W1'] = np.random.rand(n_feature,self.n_class)
        # self.model['b1'] = np.random.rand(self.n_class,)
        self.model['W1'] = np.zeros((self.n_input,self.n_class))
        self.model['b1'] = np.zeros(self.n_class,)

    def softmax(self,x):
        res = x-np.max(x)
        return np.exp(res)/np.sum(np.exp(res))

    def forward(self,x):
        z = x@self.model['W1']+self.model['b1']
        output = self.softmax(z)
        return output
    
    def predict_prob(self,x):
        z = x@self.model['W1']+self.model['b1']
        output = self.softmax(z)
        return output
    
    def predict(self,x):
        z = x@self.model['W1']+self.model['b1']
        output = self.softmax(z)
        return np.argmax(output)+1

    def score(self,X,y):
        result = [self.predict(x) for x in X]
        labels = []
        if len(X)==len(y):
            labels = y
        else:
            labels = y.reshape((-1,))
        accuracy = np.where(np.array(result)==np.array(labels),1.0,0)
        accuracy = accuracy.sum()
        accuracy /= len(result)
        return accuracy

    def fit(self,X_train,y_train,max_iter=100):
        '''
        Finds the appropriate values for W and b
        Return a list of losses in all epochs
        '''
        loss_train = []
        for i in range(max_iter):
            self.gradient_step(X_train,y_train)
            accuracy = self.score(X_train,y_train)
            loss_train.append(1-accuracy)
            if self.verbose:
                if i%50==0:
                    print('at epoch',i,'training accuracy:',accuracy)
            if accuracy>0.99:
                break
            if i>0 and i%100==0 and self.lrate>0.001:
                # self.lrate = self.lrate/2
                pass
            else:
                # self.lrate = 0.1
                pass
        if self.verbose:
            print('lrate',self.lrate)
        return loss_train

    def get_gradient(self, X_train, y_train):
        xs, hs1, hs2, errs = [], [], [], []
        for x, cls_idx in zip(X_train, y_train):
            y_pred = self.forward(x)

            # Create one-hot coding of true label
            y_true = np.zeros(self.n_class)
            y_true[int(cls_idx)-1] = 1.
            err = -y_true + y_pred
            xs.append(x)
            errs.append(err)
        return self.backward(np.array(xs),np.array(errs))

    def backward(self,xs, errs):
        dW1 = (xs.T @ errs) / xs.shape[0]
        db1 = np.sum(errs, axis=0) / xs.shape[0]
        return {'W1':dW1,'b1':db1}

    def gradient_step(self, X_train, y_train):
        grad = self.get_gradient(X_train, y_train)
        model = self.model.copy()
        for layer in grad:
            model[layer] -= self.lrate * grad[layer]
        self.model = model
        if self.verbose:
            # print(grad['b1'])
            pass
