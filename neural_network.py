import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sequence = []
        self.weight_matrix = []
        self.bias_vector = []

    def add_hidden_layer(self, nodes):
        self.sequence.append(nodes)

    def create_weight_matrix(self):
        self.weight_matrix = []
        # input layer
        self.weight_matrix.append(np.random.rand(self.input_size, self.hidden_layer_sequence[0]))
        # hidden layer
        for i in range(len(self.sequence) - 1):
            self.weight_matrix.append(np.random.rand(self.hidden_layer_sequence[i], self.hidden_layer_sequence[i + 1]))
        # output layer
        self.weight_matrix.append(np.random.rand(self.hidden_layer_sequence[-1], self.output_size))
        
    def create_bias_vector(self):
        self.bias_vector = []
        # hidden layer
        for i in range(len(self.sequence)):
            self.bias_vector.append(np.random.rand(self.hidden_layer_sequence[i]))
        # output layer
        self.bias_vector.append(np.random.rand(self.output_size))
    
    
    def feed_forward(self, input_data):
        # input layer
        input_layer = input_data
        # hidden layer
        hidden_layer = []
        for i in range(len(self.sequence)):
            hidden_layer.append(np.dot(input_layer, self.weight_matrix[i]) + self.bias_vector[i])
            input_layer = hidden_layer[i]
        # output layer
        output_layer = np.dot(input_layer, self.weight_matrix[-1]) + self.bias_vector[-1]
        return output_layer
    
    # def bac
        
        