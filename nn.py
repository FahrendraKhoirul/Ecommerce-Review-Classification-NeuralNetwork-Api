import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # create 
        np.random.seed(seed=42) # for reproducibility5

        self.W1 = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(2 / (input_size + hidden_size))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b1 = np.random.randn(self.hidden_size, 1) 
        self.b2 = np.random.randn(self.output_size, 1) 

        # self.W1 = np.random.rand(self.hidden_size, self.input_size) - 0.5
        # self.W2 = np.random.rand(self.output_size, self.hidden_size) - 0.5
        # self.b1 = np.random.rand(self.hidden_size, 1) - 0.5
        # self.b2 = np.random.rand(self.output_size, 1) - 0.5
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        expZ = np.exp(z - np.max(z))
        return expZ / (expZ.sum(axis=0, keepdims=True) + 1e-8) 
    
    def deriv_relu(self, z):
        return np.where(z > 0, 1, 0)
    
    def forward(self, X):
        A0 = X
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        return A0, Z1, A1, Z2, A2
    
    def backward(self, X, Y, Z1, A1, Z2, A2):
        one_hot_Y = self.one_hot_encode(Y)
        m = X.shape[1]
        # print("X: ", X.shape)
        # print("m: ", m)
        dZ2 = A2 - one_hot_Y
        # self.print_info(dZ2, "dZ2")
        dW2 = 1/m * dZ2.dot(A1.T)
        db2 = 1/m * np.sum(dZ2, 1)
        #  before => dZ1 =  self.W2.T.dot(dZ2) * self.deriv_relu(Z1)
        dZ1 =  self.W2.T.dot(dZ2) * self.deriv_relu(Z1) 
        # dZ1 = self.deriv_relu(Z1) * self.W2.T.dot(dZ2) 
        dW1 = 1/m * dZ1.dot(X.T)
        db1 = 1/m * np.sum(dZ1, 1)
        return dW1, db1, dW2, db2
    
    def update_weight_bias(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        # self.print_info(self.b1, "self.b1")
        # self.print_info(db1, "db1")
        # self.print_info(np.reshape(db1, (self.hidden_size, 1)), "np.reshape(db1, (self.hidden_size, 1))")
        self.b1 -= learning_rate * np.reshape(db1, (self.hidden_size, 1))
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * np.reshape(db2, (self.output_size, 1))

    def train(self, X, Y, epochs, learning_rate, patience=10):
        '''Train the neural network using the given input and output data.\nPlease note the Y should be one-hot encoded.'''
        history_cost = []
        history_acc = []
        best_epoch = 0
        no_improvement_since = 0
        for epoch in range(epochs):
            _, Z1, A1, Z2, A2 = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(X, Y, Z1, A1, Z2, A2)
            self.update_weight_bias(dW1, db1, dW2, db2, learning_rate)
            cost = -np.mean(self.one_hot_encode(Y) * np.log(A2 + 1e-8))
            predictions = np.argmax(A2, 0)
            train_acc = np.mean(predictions == Y)

            history_cost.append(cost)
            history_acc.append(train_acc)

            print(f'Epoch {epoch + 1}/{epochs} - train cost: {cost:.4f}, train acc: {train_acc:.4f}')

            # if cost <= min(history_cost):  # Check for non-increasing loss
            #     best_epoch = epoch
            #     no_improvement_since = 0
            # else:
            #     no_improvement_since += 1

            # if no_improvement_since >= patience:
            #     print(f'Early stopping at epoch {epoch + 1} due to no improvement in training loss for {patience} epochs.')
            #     break
    
        return history_cost, history_acc, best_epoch

    
    def one_hot_encode(self, Y):
        '''One hot encode the given input data.'''
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def print_info(self, npObject, title):
        print("---- ", title, " ----")
        print("Shape: ", npObject.shape)
        # print("Type: ", npObject.dtype)
        print("Size: ", npObject.size)
        print("Dimension: ", npObject.ndim)
        print("isNumpy: ", isinstance(npObject, np.ndarray))
        print("Data: ", npObject)
        print("\n")

    def predict(self, X):
        '''Predict the output based on the given input data.'''
        _, _, _, _, A2 = self.forward(X)
        result = {
            "probability": A2,
            "prediction": np.argmax(A2, 0)
        }
    
    def compute_loss(self, Y, Y_hat):
        '''Compute the loss between the true output and the predicted output.'''
        # compute loss using categorical cross-entropy
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / Y.shape[1]