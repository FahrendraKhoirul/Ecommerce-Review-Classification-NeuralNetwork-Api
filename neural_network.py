import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sequence = []
        self.hidden_layer = []
        self.weight_matrix = []
        self.bias_matrix = []
        self.learning_rate = 0.001
        self.input_dataset = None
        self.target_dataset = None
        self.epochs = None
        

    def init(self):
        self.create_weight_matrix()
        self.create_bias_matrix()
        for i in range(len(self.bias_matrix)):
            self.bias_matrix[i] = np.array(self.bias_matrix[i]).reshape(-1,1)
        # print info all weight and bias
        for i in range(len(self.weight_matrix)):
            self.print_info(self.weight_matrix[i], "Weight Matrix " + str(i))
        for i in range(len(self.bias_matrix)):
            self.print_info(self.bias_matrix[i], "Bias Matrix " + str(i))

    def add_hidden_layer(self, nodes):
        self.hidden_layer_sequence.append(nodes)

    def create_weight_matrix(self):
        self.weight_matrix = []
        # input layer
        self.weight_matrix.append(np.random.rand(self.hidden_layer_sequence[0], self.input_size) * 2 - 1) # random number between -1 and 1
        # hidden layer
        for i in range(len(self.hidden_layer_sequence) - 1):
            self.weight_matrix.append(np.random.rand(self.hidden_layer_sequence[i + 1], self.hidden_layer_sequence[i]) * 2 - 1) # random number between -1 and 1
        # output layer
        self.weight_matrix.append(np.random.rand(self.output_size,self.hidden_layer_sequence[-1]) * 2 - 1) # random number between -1 and 1
        
    def create_bias_matrix(self):
        self.bias_matrix = []
        # hidden layer
        for i in range(len(self.hidden_layer_sequence)):
            self.bias_matrix.append(np.random.rand(self.hidden_layer_sequence[i]) * 2 - 1) # random number between -1 and 1
        # output layer
        self.bias_matrix.append(np.random.rand(self.output_size) * 2 - 1) # random number between -1 and 1
        
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
        # self.print_info(y_true, "True")
        # self.print_info(y_pred, "Predict")
        predict = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # make predict to be 1D array
        predict = predict.flatten()
        # self.print_info(predict, "Predict")
        # Calculate cross-entropy loss
        loss = y_true * np.log(predict)
        loss = loss.reshape(-1,1)
        # self.print_info(loss, "Loss")
        return loss

    
    def one_hot_encode(self, x):
        return np.eye(3)[x]
    
    def print_info(self, npObject, title):
        print("---- ", title, " ----")
        print("Shape: ", npObject.shape)
        # print("Type: ", npObject.dtype)
        print("Size: ", npObject.size)
        print("Dimension: ", npObject.ndim)
        print("isNumpy: ", isinstance(npObject, np.ndarray))
        print("Data: ", npObject)
        print("\n")

    # =============== Vectorization ===============
        
    def feedforward_vectorization_mode(self, tfidf_input_vector):
        self.input_data = np.array(tfidf_input_vector).reshape(-1,1)
        self.hidden_layer = []
        

        x = self.input_data
        # self.print_info(x, "Input Data")
        w1 = self.weight_matrix[0]
        # self.print_info(w1, "Weight Matrix 1")
        w2 = self.weight_matrix[-1]
        # self.print_info(w2, "Weight Matrix 2")
        b1 = self.bias_matrix[0]
        # self.print_info(b1, "Bias Matrix 1")
        b2 = self.bias_matrix[-1]
        # self.print_info(b2, "Bias Matrix 2")

        # Hidden Layer 1
        z1 = np.dot(w1, x) + b1
        # self.print_info(z1, "Hidden Layer 1")
        a1 = self.sigmoid(z1)
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
        # self.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               lf.derivative_softmax(output_layer), "der Output Layer")
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

        # 𝑑𝐿𝑑𝑊= 𝛿1𝑥
        dw1 = np.dot(delta_1, self.input_data.T)
        # self.print_info(dw1, "Gradient Weight 1")

        # 𝑑𝐿𝑑𝑏= 𝛿1
        db1 = delta_1
        # self.print_info(db1, "Gradient Bias 1")

        #========= UPDATE WEIGHT AND BIAS ==================
        # self.print_info(self.weight_matrix[-1], "Weight Matrix 2")
        # self.print_info(dw2, "Gradient Weight 2")
        # self.print_info(self.bias_matrix[-1], "Bias Matrix 2")
        # self.print_info(db2, "Gradient Bias 2")
        # self.print_info(self.weight_matrix[-2], "Weight Matrix 1")
        # self.print_info(dw1, "Gradient Weight 1")
        # self.print_info(self.bias_matrix[-2], "Bias Matrix 1")
        # self.print_info(db1, "Gradient Bias 1")
        # self.weight_matrix[-1] -=  dw2 * self.learning_rate
        # self.bias_matrix[-1] = np.array(self.bias_matrix[-1]) - db2 * self.learning_rate
        # self.weight_matrix[-2] -= dw1 * self.learning_rate
        # self.bias_matrix[-2] = np.array(self.bias_matrix[-2]) - db1 * self.learning_rate
        
        sum_error = -np.sum(error)
        return sum_error, dw2, db2, dw1, db1

    def train_vectorization_mode(self, input_dataset, target_data, epochs, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input_data = input_dataset
        self.target_data = self.one_hot_encode(target_data)
        print("Input Data: ", self.input_data)
        print("Target Data: ", self.target_data)
        self.epochs = epochs

        history = []

        for epoch in range(epochs):
            output_layer = self.feedforward_vectorization_mode(input_dataset)
            error = self.backpropagation_vectorization_mode(target_data, output_layer)
            history.append(error)
            print("Epoch ", epoch, "Error: ", error)
            
        return history
    
    def train(self, input_dataset, target_dataset, epochs, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input_data = input_dataset
        self.target_data = target_dataset
        self.epochs = epochs

        history = []
        total_data = len(input_dataset)
        total_error = 0
        
        # create copy of weight and bias with zero value
        dw1 = np.zeros(self.weight_matrix[0].shape)
        db1 = np.zeros(self.bias_matrix[0].shape)
        dw2 = np.zeros(self.weight_matrix[-1].shape)
        db2 = np.zeros(self.bias_matrix[-1].shape)

        # for epoch in range(epochs):
        for epoch in range(epochs):
            for i in range(input_dataset.shape[0]):
                output_layer = self.feedforward_vectorization_mode(input_dataset.iloc[i])
                backprop_error, backprop_dw2, backprop_db2, backprop_dw1, backprop_db1 = self.backpropagation_vectorization_mode(target_dataset.iloc[i], output_layer)
                
                # sum error, weight, and bias
                total_error += backprop_error
                dw2 += backprop_dw2
                db2 += backprop_db2
                dw1 += backprop_dw1
                db1 += backprop_db1

            # calculate average error
            total_error = total_error / total_data
            history.append(total_error)
            print("Epoch ", epoch, "Error: ", total_error)
            total_error = 0

             # Update Weight and Bias
            dw2 = dw2 / total_data
            db2 = db2 / total_data
            dw1 = dw1 / total_data
            db1 = db1 / total_data

            self.weight_matrix[-1] -=  dw2 * self.learning_rate
            self.bias_matrix[-1] = np.array(self.bias_matrix[-1]) - db2 * self.learning_rate
            self.weight_matrix[-2] -= dw1 * self.learning_rate
            self.bias_matrix[-2] = np.array(self.bias_matrix[-2]) - db1 * self.learning_rate

        return history
