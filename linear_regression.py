import numpy as np

class LinearRegression():

    def __init__(self, learning_rate=0.0001, epochs=10000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []  
    
    def _compute_loss(self, y, y_pred):
        # Mean Squared Error Loss
        return np.mean((y - y_pred) ** 2)

    def compute_gradients(self, x, y, y_pred):
        # Compute gradients for weights and bias
        grad_w = -2 * np.dot(x.T, (y - y_pred)) / y.size
        grad_b = -2 * np.mean(y - y_pred)
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def fit(self, X, y):
  
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        X = X.values.reshape(-1, 1) if len(X.shape) == 1 else X
        y = y.values

        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        # Gradient Descent
        for _ in range(self.epochs):
            y_pred = X.dot(self.weights) + self.bias   # (n,)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

        return self.weights, self.bias

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        X = X.values.reshape(-1, 1) if len(X.shape) == 1 else X
        return X.dot(self.weights) + self.bias
