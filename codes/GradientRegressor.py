import numpy as np

# @ Implementing Gradient Descent
class GDRegressor:
    def __init__(self,learning_rate,epochs):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,X,y):
        # Calculate the b using Gradient Descent
        for _ in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            self.b = self.b - self.lr * loss_slope_b
            # Calculate the m using Gradient Descent
            loss_slope_m = -2 * np.sum(X.ravel() * (y - self.m * X.ravel() - self.b))
            self.m = self.m - self.lr * loss_slope_m

    def predict(self,X):
        y_pred = self.m * X.ravel() + self.b
        return y_pred
    
    def score(self,X,y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def get_params(self):
        return self.m,self.b
    
    def set_params(self,m,b):
        self.m = m
        self.b = b



# @ Implementing Stochastic Gradient Descent
class SGDRegressor:
    def __init__(self,learning_rate,epochs):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,X,y):
        # Calculate the b using Gradient Descent
        for _ in range(self.epochs):
            for i in range(len(X)):
                loss_slope_b = -2 * (y[i] - self.m * X[i] - self.b)
                self.b = self.b - self.lr * loss_slope_b
                # Calculate the m using Gradient Descent
                loss_slope_m = -2 * X[i] * (y[i] - self.m * X[i] - self.b)
                self.m = self.m - self.lr * loss_slope_m

    def predict(self,X):
        y_pred = self.m * X.ravel() + self.b
        return y_pred
    
    def score(self,X,y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def get_params(self):
        return self.m,self.b
    
    def set_params(self,m,b):
        self.m = m
        self.b = b

# @ Implementing Mini Batch Gradient Descent
class MiniBatchGDRegressor:
    def __init__(self,learning_rate,epochs,batch_size):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self,X,y):
        # Calculate the b using Gradient Descent
        for _ in range(self.epochs):
            for i in range(0,len(X),self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                loss_slope_b = -2 * np.sum(y_batch - self.m * X_batch.ravel() - self.b)
                self.b = self.b - self.lr * loss_slope_b
                # Calculate the m using Gradient Descent
                loss_slope_m = -2 * np.sum(X_batch.ravel() * (y_batch - self.m * X_batch.ravel() - self.b))
                self.m = self.m - self.lr * loss_slope_m

    def predict(self,X):
        y_pred = self.m * X.ravel() + self.b
        return y_pred
    
    def score(self,X,y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def get_params(self):
        return self.m,self.b
    
    def set_params(self,m,b):
        self.m = m
        self.b = b


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    import matplotlib.pyplot as plt
    X,y = make_regression(n_samples=1000,n_features=1,n_informative=1,noise=10,random_state=1)
    X = X.ravel()
    y = y.ravel()
    plt.scatter(X,y)
    plt.show()
    # Gradient Descent
    model = GDRegressor(learning_rate=0.0001,epochs=100)
    model.fit(X,y)
    y_pred = model.predict(X)
    plt.scatter(X,y)
    plt.plot(X,y_pred,color='red')
    plt.show()
    print('Score:',model.score(X,y))
    # Stochastic Gradient Descent
    model = SGDRegressor(learning_rate=0.0001,epochs=100)
    model.fit(X,y)
    y_pred = model.predict(X)
    plt.scatter(X,y)
    plt.plot(X,y_pred,color='red')
    plt.show()
    print('Score:',model.score(X,y))
    # Mini Batch Gradient Descent
    model = MiniBatchGDRegressor(learning_rate=0.0001,epochs=100,batch_size=10)
    model.fit(X,y)
    y_pred = model.predict(X)
    plt.scatter(X,y)
    plt.plot(X,y_pred,color='red')
    plt.show()
    print('Score:',model.score(X,y))

    


