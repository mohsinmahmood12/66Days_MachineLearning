# @ Implementation of Preceptron

import numpy as np

class Perceptron:
    def __init__(self,learning_rate=0.2,n_iters =1000):
        self.lr = learning_rate
        self.epoch = n_iters
        self.weights = None 
        self.bias = None 

    def fit(self,X,y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epoch):
            for i in range(X.shape[0]):
                y_pred = np.dot(self.weights,X[i]) + self.bias 
                self.weights = self.weights + self.lr * (y[i] - y_pred) * X[i]
                self.bias = self.bias + self.lr * (y[i] - y_pred)
        
        print("|---Training Completed---|")

    # @ Calculating activation function
    def activation_function(self,activation):
        if activation >= 0:
            return 1
        else:
            return 0
        
    def predict(self,X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.activation_function(np.dot(self.weights,X[i]) + self.bias))
        return np.array(y_pred)
    
    def accuracy(self,y_true,y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
# @ Load the data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
# @ Split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
# @ Train the model
perceptron = Perceptron(learning_rate=0.2,n_iters=1000)
perceptron.fit(X_train,y_train)
# @ Predict the model
predictions = perceptron.predict(X_test)
# @ Accuracy
accuracy = perceptron.accuracy(y_test,predictions)
print("Accuracy:",accuracy)
