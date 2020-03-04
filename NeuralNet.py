import numpy as np

class NeuralNet():
    def __init__(self,n_feature,n_class,lrate = 0.1):
        self.n_input = n_feature
        self.n_class = n_class
        self.n_hidden1 = 500
        self.n_hidden2 = 500
        self.lrate = lrate
        self.model = {}

        self.model['W1'] = np.random.rand(n_feature,self.n_hidden1)
        self.model['W2'] = np.random.rand(self.n_hidden1,self.n_hidden2)
        self.model['W3'] = np.random.rand(self.n_hidden2,self.n_class)

        self.model['b1'] = np.random.rand(self.n_hidden1,)
        self.model['b2'] = np.random.rand(self.n_hidden2,)
        self.model['b3'] = np.random.rand(self.n_class,)

    def softmax(self,x):
        res = x-np.max(x)
        return np.exp(res)/np.sum(np.exp(res))

    def forward(self,x):
        z1 = x@self.model['W1']+self.model['b1']
        h1 = z1
        h1[z1<0] = 0
        z2 = h1@self.model['W2']+self.model['b2']
        h2 = z2
        h2[z2<0] = 0
        z = h2@self.model['W3']+self.model['b3']
        output = self.softmax(z)
        return h1,h2,output
    
    def predict_prob(self,x):
        z1 = x@self.model['W1']+self.model['b1']
        h1 = z1
        h1[z1<0] = 0
        z2 = h1@self.model['W2']+self.model['b2']
        h2 = z2
        h2[z2<0] = 0
        z = h2@self.model['W3']+self.model['b3']
        output = self.softmax(z)
        return output
    
    def predict(self,x):
        z1 = x@self.model['W1']+self.model['b1']
        h1 = z1
        h1[z1<0] = 0
        z2 = h1@self.model['W2']+self.model['b2']
        h2 = z2
        h2[z2<0] = 0
        z = h2@self.model['W3']+self.model['b3']
        output = self.softmax(z)
        return np.argmax(output)+1

    def fit(self,X_train,y_train,max_iter=100):
        for i in range(max_iter):
            self.gradient_step(X_train,y_train)
            result = [self.predict(x) for x in X_train]
            labels = y_train.reshape((-1,))
            accuracy = np.where(result==labels,1,0)
            # print(accuracy)
            accuracy = accuracy.sum()
            accuracy /= len(result)
            print('at epoch',i,'training accuracy:',accuracy)
            if accuracy>0.99:
                break
            if i>0 and i%100==0 and self.lrate>0.00001 and False:
                self.lrate = self.lrate/2
            else:
                self.lrate = 0.1
        print('lrate',self.lrate)

    def get_gradient(self, X_train, y_train):
        xs, hs1, hs2, errs = [], [], [], []
        for x, cls_idx in zip(X_train, y_train):
            h_1, h_2, y_pred = self.forward(x)

            # Create one-hot coding of true label
            y_true = np.zeros(self.n_class)
            y_true[int(cls_idx)-1] = 1.
            # y_true[int(cls_idx)] = 1.
            # print(y_true)

            # Compute the gradient of output layer
            err = -y_true + y_pred

            # Accumulate the informations of the examples
            # x: input
            # h: hidden state
            # err: gradient of output layer
            xs.append(x)
            hs1.append(h_1)
            hs2.append(h_2)
            errs.append(err)
        # print('error',errs)
        return self.backward(np.array(xs),np.array(hs1),np.array(hs2),np.array(errs))

    def backward(self,xs, hs1, hs2, errs):
        # errs is the gradients of output layer for the minibatch
        # print("errs %r"%str(np.shape(errs)))
        dW3 = (hs2.T @ errs) / xs.shape[0]
        # print('dW3',np.shape(dW3))

        # Get gradient of hidden layer
        dh2 = errs @ self.model['W3'].T
        dh2[hs2 <= 0] = 0
        dW2 = (hs1.T @ dh2) / xs.shape[0]

        dh1 = (dh2 @ self.model['W2'].T) / xs.shape[0]
        dh1[hs1 <= 0] = 0
        dW1 = (xs.T @ dh1) / xs.shape[0]

        # Get gradient for the bias of hidden layer
        # dmodel['b1'] = np.sum(dh,axis=0)/ xs.shape[0]
        # dmodel['b2'] = np.sum(errs,axis=0)/ xs.shape[0]
        db1 = np.sum(dh1, axis=0) / xs.shape[0]
        db2 = np.sum(dh2, axis=0) / xs.shape[0]
        db3 = np.sum(errs, axis=0) / xs.shape[0]
        return {'W1':dW1,'W2':dW2,'W3':dW3,'b1':db1,'b2':db2,'b3':db3}

    def gradient_step(self, X_train, y_train):
        grad = self.get_gradient(X_train, y_train)
        model = self.model.copy()
        for layer in grad:
            model[layer] -= self.lrate * grad[layer]
        self.model = model
