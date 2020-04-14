import numpy as np

class NeuralNet():
    def __init__(self,n_feature,n_class,lrate = 0.1,verbose=False):
        self.verbose = verbose
        self.n_input = n_feature
        self.n_class = n_class
        self.n_hidden1 = 500
        self.lrate = lrate
        self.model = {}

        self.model['W1'] = np.random.rand(n_feature,self.n_hidden1)
        self.model['W2'] = np.random.rand(self.n_hidden1,self.n_class)

        self.model['b1'] = np.random.rand(self.n_hidden1,)
        self.model['b2'] = np.random.rand(self.n_class,)

    def softmax(self,x):
        res = x-np.max(x)
        return np.exp(res)/np.sum(np.exp(res))

    def forward(self,x):
        z1 = x@self.model['W1']+self.model['b1']
        h1 = z1
        h1[z1<0] = 0
        z = h1@self.model['W2']+self.model['b2']
        output = self.softmax(z)
        return h1,output
    
    def predict_prob(self,x):
        z1 = x@self.model['W1']+self.model['b1']
        h1 = z1
        h1[z1<0] = 0
        z = h1@self.model['W2']+self.model['b2']
        output = self.softmax(z)
        return output
    
    def predict(self,x):
        z1 = x@self.model['W1']+self.model['b1']
        h1 = z1
        h1[z1<0] = 0
        z = h1@self.model['W2']+self.model['b2']
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
            # if i>0 and i%100==0 and self.lrate>0.001:
            #     self.lrate = self.lrate/2
            # else:
            #     self.lrate = 0.1
        if self.verbose:
            print('lrate',self.lrate)
        return loss_train

    def get_gradient(self, X_train, y_train):
        xs, hs1, hs2, errs = [], [], [], []
        for x, cls_idx in zip(X_train, y_train):
            h_1, y_pred = self.forward(x)

            # Create one-hot coding of true label
            y_true = np.zeros(self.n_class)
            y_true[int(cls_idx)-1] = 1.
            # Compute the gradient of output layer
            err = -y_true + y_pred

            xs.append(x)
            hs1.append(h_1)
            errs.append(err)
        # print('error',errs)
        return self.backward(np.array(xs),np.array(hs1),np.array(errs))

    def backward(self,xs, hs1, errs):
        # print("errs %r"%str(np.shape(errs)))
        dW2 = (hs1.T @ errs) / xs.shape[0]
        # print('dW3',np.shape(dW3))

        # Get gradient of hidden layer
        dh1 = (errs @ self.model['W2'].T) / xs.shape[0]
        dh1[hs1 <= 0] = 0
        dW1 = (xs.T @ dh1) / xs.shape[0]

        # Get gradient for the bias of hidden layer
        db1 = np.sum(dh1, axis=0) / xs.shape[0]
        db2 = np.sum(errs, axis=0) / xs.shape[0]
        return {'W1':dW1,'W2':dW2,'b1':db1,'b2':db2}

    def gradient_step(self, X_train, y_train):
        grad = self.get_gradient(X_train, y_train)
        model = self.model.copy()
        for layer in grad:
            model[layer] -= self.lrate * grad[layer]
        self.model = model
