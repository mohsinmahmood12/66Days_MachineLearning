import numpy as np
from sklearn.datasets import load_iris
class LogisticRegression:
    # @Declare learning rate and number of iterations
    def __init__(self,learning_rate=0.001,n_iters=1000):
        self.lr = learning_rate
        self.n_iters= n_iters
        self.weights = None
        self.bias = None
    # @fit function to train the model with dataset
    def fit(self,X,y):
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features) # Initialize Weights
        self.bias = 0                       # Initialize Bias
        # @ Gradient Descent
        for i in range(self.n_iters):
            linear_model = np.dot(X,self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            # Computing the Gradient
            dw = (1 / n_samples) * np.dot(X.T,(y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # now updating the parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 

    # Here Get the Test Sample that we wants to predict
    def predict(self,X):
        # applying linear model
        linear_model = np.dot(X,self.weights) + self.bias
        # define the predict model
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    #@ Sigmoid Function
    def sigmoid(self,x):
        return 1 / ( 1 + np.exp(-x))
    

# Load iris data
iris = load_iris()
X = iris.data
y = iris.target
# @Split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
# @Train the model
regressor = LogisticRegression(learning_rate=0.001,n_iters=1000)
regressor.fit(X_train,y_train)
# @Predict the model
predictions = regressor.predict(X_test)
# @Accuracy
def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print("Logistic Regression Classification Accuracy",accuracy(y_test,predictions))


