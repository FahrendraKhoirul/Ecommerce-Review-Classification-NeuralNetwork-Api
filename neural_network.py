import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sequence = []
        self.hidden_layer = []
        self.weight_matrix = []
        self.bias_matrix = []
        self.learning_rate = 0.1
        self.input_data = None
        self.target_data = None
        self.epochs = None

    def add_hidden_layer(self, nodes):
        self.hidden_layer_sequence.append(nodes)

    def create_weight_matrix(self):
        self.weight_matrix = []
        # input layer
        self.weight_matrix.append(np.random.rand(self.hidden_layer_sequence[0], self.input_size))
        # hidden layer
        for i in range(len(self.hidden_layer_sequence) - 1):
            self.weight_matrix.append(np.random.rand(self.hidden_layer_sequence[i + 1], self.hidden_layer_sequence[i]))
        # output layer
        self.weight_matrix.append(np.random.rand(self.output_size,self.hidden_layer_sequence[-1]))
        
    def create_bias_matrix(self):
        self.bias_matrix = []
        # hidden layer
        for i in range(len(self.hidden_layer_sequence)):
            self.bias_matrix.append(np.random.rand(self.hidden_layer_sequence[i]))
        # output layer
        self.bias_matrix.append(np.random.rand(self.output_size))
    
    
    def feed_forward(self, input_data):
        # input layer
        input_layer = np.array(input_data)
        self.input_data = input_layer
        # hidden layer
        self.hidden_layer = []
        for i in range(len(self.hidden_layer_sequence)):
            self.hidden_layer.append(np.dot(input_layer, self.weight_matrix[i]) + self.bias_matrix[i])
            self.hidden_layer[i] = self.relu(self.hidden_layer[i])
            input_layer = self.hidden_layer[i]
        # output layer
        output_layer = np.dot(input_layer, self.weight_matrix[-1]) + self.bias_matrix[-1]
        output_layer = self.softmax(output_layer)
        # self.print_info(output_layer, "Output Layer")
        return output_layer
    

    def backpropagation(self, target_data, output_layer):
        # calculate error (TEORY)
        # error = target_data - output_layer # [1,0,0] - [0.7,0.2,0.1] = [0.3,-0.2,-0.1]

        # calculate error (Cross Entropy)
        error = self.categorical_crossentropy_loss(target_data, output_layer)
        # self.print_info(error, "Error")

        #========  OUTPUT LAYER (Hidden Layer 2 in my Skripsi)  ==================
        # 𝛿2=(𝑦−ℎ2)𝜎′(ℎ2)
        # self.print_info(output_layer, "Output Layer")
        delta_2 = error * self.derivative_softmax(output_layer.reshape(-1,1)) 
        # self.print_info(delta_2, "Delta 2")


        # 𝑑𝐿𝑑𝑉= 𝛿2(ℎ1)𝑇
        # self.print_info(self.hidden_layer[-1], "Hidden Layer 2")
        gradient_weight_2 = np.dot(self.hidden_layer[-1].reshape(-1,1), delta_2.T)
        # self.print_info(gradient_weight_2, "Gradient Weight 2")

        # 𝑑𝐿𝑑𝑏= 𝛿2
        gradient_bias_2 = delta_2
        # self.print_info(gradient_bias_2, "Gradient Bias 2")

        #========  HIDDEN LAYER (Hidden Layer 1 in my Skripsi)  ==================
        # 𝛿1= 𝑉𝑇∗𝛿2∗𝜎′(𝑎1)
        self.print_info(self.weight_matrix[-1], "Weight Matrix 2")
        # self.print_info(self.hidden_layer[-1], "Hidden Layer 2")
        delta_1 = np.dot(self.weight_matrix[-1], delta_2) 
        self.print_info(delta_1, "Delta 1 sebelum relu")
        self.print_info(self.derivative_relu(self.hidden_layer[-1].reshape(-1,1)), "Derivative Relu")
        delta_1 = np.dot(delta_1.T, self.derivative_relu(self.hidden_layer[-1].reshape(-1,1)))
        self.print_info(delta_1, "Delta 1 setelah relu")

        # 𝑑𝐿𝑑𝑊= 𝛿1𝑥
        gradient_weight_1 = np.dot(self.input_data.reshape(-1, 1), delta_1)

        # 𝑑𝐿𝑑𝑏= 𝛿1
        gradient_bias_1 = delta_1

        #========= UPDATE WEIGHT AND BIAS ==================
        self.weight_matrix[-1] += gradient_weight_2 * self.learning_rate
        self.bias_matrix[-1] += gradient_bias_2 * self.learning_rate
        self.weight_matrix[-2] += gradient_weight_1 * self.learning_rate
        self.bias_matrix[-2] += gradient_bias_1 * self.learning_rate
        
        return error
    
    def train(self, input_data, target_data, epochs):
        self.input_data = input_data
        self.target_data = target_data
        self.epochs = epochs

        for epoch in range(epochs):
            output_layer = self.feed_forward(input_data)
            error = self.backpropagation(target_data, output_layer)
            print("Epoch ", epoch, "Error: ", error)
        
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0, keepdims=True)
    
    def derivative_relu(self, x):
        return 1 * (x > 0)
    
    def derivative_softmax(self, x):
        return x * (1 - x)
    
    def categorical_crossentropy_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))
    
    def one_hot_encode(self, x):
        return np.eye(3)[x]
    
    def print_info(self, npObject, title):
        print("---- ", title, " ----")
        print("Shape: ", npObject.shape)
        # print("Type: ", npObject.dtype)
        print("Size: ", npObject.size)
        print("Dimension: ", npObject.ndim)
        print("isNumpy: ", isinstance(npObject, np.ndarray))
        # print("Data: ", npObject)
        print("\n")

    # =============== Vectorization ===============
        
    def feedforward_vectorization_mode(self, tfidf_input_vector):
        self.input_data = np.array(tfidf_input_vector).reshape(-1,1)
        self.hidden_layer = []
        

        x = np.array(tfidf_input_vector).reshape(-1,1)
        # self.print_info(x, "Input Data")
        w1 = self.weight_matrix[0]
        # self.print_info(w1, "Weight Matrix 1")
        w2 = self.weight_matrix[-1]
        # self.print_info(w2, "Weight Matrix 2")
        b1 = np.array(self.bias_matrix[0]).reshape(-1,1)
        # self.print_info(b1, "Bias Matrix 1")
        b2 = np.array(self.bias_matrix[-1]).reshape(-1,1)
        # self.print_info(b2, "Bias Matrix 2")

        # Hidden Layer 1
        z1 = np.dot(w1, x) + b1
        # self.print_info(z1, "Hidden Layer 1")
        a1 = self.relu(z1)
        self.hidden_layer.append(a1)
        # self.print_info(a1, "Relu Hidden Layer 1")

        # Output Layer
        z2 = np.dot(w2, a1) + b2
        # self.print_info(z2, "Output Layer")
        a2 = self.softmax(z2)
        self.hidden_layer.append(a2)
        # self.print_info(a2, "Softmax Output Layer")

        output = a2
        return output
    
    def backpropagation_vectorization_mode(self, target_data, output_layer):
        # calculate error (Cross Entropy)
        error = self.categorical_crossentropy_loss(target_data, output_layer)
        # self.print_info(error, "Error")

        #========  OUTPUT LAYER (Hidden Layer 2 in my Skripsi)  ==================
        # 𝛿2=(𝑦−ℎ2)𝜎′(ℎ2)
        delta_2 = error * self.derivative_softmax(output_layer)
        # self.print_info(delta_2, "Delta 2")

        # 𝑑𝐿𝑑𝑉= 𝛿2(ℎ1)𝑇
        # self.print_info(self.hidden_layer[0], "Hidden Layer 1")
        dw2 = np.dot(delta_2, self.hidden_layer[0].T)
        # self.print_info(dw2, "Gradient Weight 2")
        
        # 𝑑𝐿𝑑𝑏= 𝛿2
        db2 = delta_2
        # self.print_info(db2, "Gradient Bias 2")

        #========  HIDDEN LAYER (Hidden Layer 1 in my Skripsi)  ==================
        # 𝛿1= ∗𝛿𝑉𝑇2∗𝜎′(𝑎1)
        # self.print_info(self.weight_matrix[-1], "Weight Matrix 2")
        # self.print_info(self.hidden_layer[-1], "Hidden Layer 2")
        delta_1 = np.dot( self.weight_matrix[-1].T, delta_2)
        # self.print_info(delta_1, "Delta 1 sebelum relu")
        # self.print_info(self.derivative_relu(self.hidden_layer[0]), "Derivative Relu Hidden Layer 1")

        delta_1 = delta_1 * self.derivative_relu(self.hidden_layer[0])
        # self.print_info(delta_1, "Delta 1 setelah relu")

        # 𝑑𝐿𝑑𝑊= 𝛿1𝑥T
        dw1 = np.dot(delta_1, self.input_data.T)
        # self.print_info(dw1, "Gradient Weight 1")

        # 𝑑𝐿𝑑𝑏= 𝛿1
        db1 = delta_1
        # self.print_info(db1, "Gradient Bias 1")

        #========= UPDATE WEIGHT AND BIAS ==================
        # self.print_info(self.weight_matrix[-1], "Weight Matrix 2")
        # self.print_info(dw2, "Gradient Weight 2")
        self.print_info(self.bias_matrix[-1], "Bias Matrix 2")
        self.print_info(db2, "Gradient Bias 2")
        # self.print_info(self.weight_matrix[-2], "Weight Matrix 1")
        # self.print_info(dw1, "Gradient Weight 1")
        self.print_info(self.bias_matrix[-2], "Bias Matrix 1")
        self.print_info(db1, "Gradient Bias 1")
        self.weight_matrix[-1] -=  dw2 * self.learning_rate
        self.bias_matrix[-1] = np.array(self.bias_matrix[-2]) - db2 * self.learning_rate
        self.weight_matrix[-2] -= dw1 * self.learning_rate
        self.bias_matrix[-2] = np.array(self.bias_matrix[-2]) - db1 * self.learning_rate
        
        return error

    def train_vectorization_mode(self, input_data, target_data, epochs):
        self.input_data = input_data
        self.target_data = self.one_hot_encode(target_data)
        self.epochs = epochs

        for epoch in range(epochs):
            output_layer = self.feedforward_vectorization_mode(input_data)
            error = self.backpropagation_vectorization_mode(target_data, output_layer)
            print("Epoch ", epoch, "Error: ", error)