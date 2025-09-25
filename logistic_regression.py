import numpy as np

class LogisticRegression():

    def __init__(self, learning_rate=0.0001, epochs=10000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []  
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        # Binary Cross-Entropy Loss
        eps = 1e-15  # to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def compute_gradients(self, x, y, y_pred):
        # Gradients for logistic regression
        error = y_pred - y
        grad_w = np.dot(x.T, error) / y.size
        grad_b = np.mean(error)
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
            linear = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(linear)   # logistic regression prediction
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

        return self.weights, self.bias

    def predict(self, X):
        """
        Generates probabilities
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats (probabilities)
        """
        # TODO: Implement
        X = X.values.reshape(-1, 1) if len(X.shape) == 1 else X
        linear = X.dot(self.weights) + self.bias
        return self._sigmoid(linear)

    def predict_classes(self, X):
        """
        Generates binary class predictions (0 or 1)
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of integers (0 or 1)
        """
        probs = self.predict(X)
        return (probs >= 0.5).astype(int)
