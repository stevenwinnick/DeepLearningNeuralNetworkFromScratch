"""
A deep neural network I built from scratch, with NumPy as the only imported library (for succinct matrix operations).
This was built loosely following the (unfinished) sentdex Neural Networks from Scratch YouTube Series, found here:
https://youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3 nnfs.io
The MyNeuralNet class takes a list of layer sizes (heights) as an input, and creates hidden layers and an output layer
of the appropriate sizes. It trains on batches of data of any size, arranged as a list of input vectors, with the last
[output size] elements of the input vectors containing their actual classification data for supervised learning.
"""

import numpy as np


class MyNeuralNet:

    def __init__(self, layer_sizes):
        if(len(layer_sizes)) < 3:
            print("Need hidden layers!")
            exit()
        self.input_size = layer_sizes[0]  # Number of parameters per input
        self.current_input_batch = None  # The bath of inputs itself
        self.current_batch_size = None  # Number of inputs in the batch
        self.hidden_layers = []  # Not a numpy array, since it stores objects of a custom class
        for layer_size in layer_sizes[1:-1]:
            self.hidden_layers.append(MyHiddenLayer(layer_size))
            if len(self.hidden_layers) >= 2:
                self.hidden_layers[-1].prev_layer = self.hidden_layers[-2]
                self.hidden_layers[-2].next_layer = self.hidden_layers[-1]
        self.output_size = layer_sizes[-1]
        self.output_layer = MyOutputLayer(self.output_size)
        self.output_layer.prev_layer = self.hidden_layers[-1]
        self.hidden_layers[-1].next_layer = self.output_layer
        self.current_loss = 0
        
    def predict(self, data_without_actual):

        input_batch = np.array(data_without_actual)
        current_layer_output_batch = input_batch
        for layer in self.hidden_layers:
            layer.forward_propagation(current_layer_output_batch)
            current_layer_output_batch = layer.current_output_batch
        self.output_layer.make_predictions(current_layer_output_batch)
        return self.output_layer.current_prediction_batch

    def train(self, data_batch):
       
        self.current_input_batch = np.array(data_batch)
        self.current_batch_size = np.size(self.current_input_batch, axis=0)  # Number of inputs in the batch
        
        # Separate input data into its inputs and outputs
        current_batch_data_to_pass = []
        current_batch_data_to_check = []
        for input_line in data_batch:
            current_batch_data_to_pass.append(input_line[:-self.output_size])
            current_batch_data_to_check.append(input_line[-self.output_size:])
        current_batch_data_to_pass = np.array(current_batch_data_to_pass)
        current_batch_data_to_check = np.array(current_batch_data_to_check)

        # Forward propagation
        clipped_predictions = np.clip(self.predict(current_batch_data_to_pass), 1e-7, 1 - 1e-7)

        # Calculate loss using categorical cross entropy
        products = np.multiply(-np.log(clipped_predictions), current_batch_data_to_check)
        self.current_loss = 0
        for output_row in products:
            self.current_loss += np.sum(output_row)
        # print("Loss: ", self.current_loss)

        # Calculate partial of loss wrt to previous layer's nodes
        d_loss_d_predicted = -np.divide(current_batch_data_to_check, np.log(clipped_predictions))
        d_loss_d_predicted = np.average(d_loss_d_predicted, axis=0)  # Average per output node

        # Calculate other partial derivatives for back propagation
        current_layer_derivatives = self.output_layer.get_back_prop_derivatives(d_loss_d_predicted)
        for layer in reversed(self.hidden_layers):
            current_layer_derivatives = layer.get_back_prop_derivatives(current_layer_derivatives)

        # Apply back propagation
        learning_rate = .01
        for layer in self.hidden_layers:
            layer.apply_back_propagation(self.current_loss, learning_rate)
        self.output_layer.apply_back_propagation(self.current_loss, learning_rate)


class MyHiddenLayer:

    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.prev_layer = None
        self.next_layer = None
        self.weights_and_biases_initialized = False
        self.weights = None
        self.biases = None
        self.inputs_to_layer = None
        self.current_output_batch = None
        self.all_pre_sigmoid_values = None
        self.partials_wrt_weights = None
        self.partials_wrt_biases = None
    
    @staticmethod
    def apply_activation_function(whole_batch_hidden_layer):
        new_whole_batch_hidden_layer = whole_batch_hidden_layer
        for layer in new_whole_batch_hidden_layer:
            for index, value in enumerate(layer):
                layer[index] = np.divide(1, (1 + np.exp(-value)))
        return new_whole_batch_hidden_layer
    
    def forward_propagation(self, previous_nodes_batch):
        
        if not self.weights_and_biases_initialized:
            if self.prev_layer is None:
                self.weights = np.random.rand(self.layer_size, len(previous_nodes_batch[0]))
            else:
                self.weights = np.random.rand(self.layer_size, self.prev_layer.layer_size)
            self.weights = (self.weights - 0.5) * 2
            self.biases = np.random.rand(self.layer_size)
            self.biases = (self.biases - 0.5) * 2
            self.weights_and_biases_initialized = True

        self.inputs_to_layer = previous_nodes_batch
        self.current_output_batch = \
            np.transpose(np.dot(self.weights, np.transpose(self.inputs_to_layer)))  # multiply weights by previous layer
        self.current_output_batch = np.add(self.current_output_batch, self.biases)  # add biases to each row
        self.all_pre_sigmoid_values = self.current_output_batch  # used during backpropagation
        self.current_output_batch = self.apply_activation_function(self.current_output_batch)

        return self.current_output_batch

    def get_back_prop_derivatives(self, d_loss_wrt_these_nodes):
        
        # Get partial nodes / partial pre sigmoid
        partial_nodes_partial_activation = np.empty((0, self.layer_size))
        for layer_from_an_input_in_batch in self.all_pre_sigmoid_values:
            current_layer_partials = []
            for node_pre_sigmoid in layer_from_an_input_in_batch:
                current_layer_partials.append(np.multiply(node_pre_sigmoid, np.add(1, -node_pre_sigmoid)))
            partial_nodes_partial_activation = np.append(partial_nodes_partial_activation, [current_layer_partials],
                                                         axis=0)
        partial_nodes_partial_activation = np.sum(partial_nodes_partial_activation, axis=0)  # Get total partial by node

        # Get partial of loss wrt weights into output nodes
        partial_pre_activation_partial_weights = np.array([np.sum(self.inputs_to_layer, axis=0)])
        self.partials_wrt_weights = np.repeat(partial_pre_activation_partial_weights, self.layer_size, axis=0)
        for index, node in enumerate(self.partials_wrt_weights):
            self.partials_wrt_weights[index] = np.multiply(node, np.multiply(d_loss_wrt_these_nodes[index],
                                                                             partial_nodes_partial_activation[index]))

        # Get partial of loss wrt bias into output nodes
        self.partials_wrt_biases = np.multiply(d_loss_wrt_these_nodes, partial_nodes_partial_activation)

        # Return partials wrt previous layer nodes
        return np.dot(np.multiply(d_loss_wrt_these_nodes, partial_nodes_partial_activation), self.weights)

    def apply_back_propagation(self, total_loss, learning_rate):
        self.weights = np.add(self.weights, np.multiply(self.partials_wrt_weights,
                                                        np.multiply(total_loss, learning_rate)))
        self.biases = np.add(self.biases, np.multiply(self.partials_wrt_biases, np.multiply(total_loss, learning_rate)))


class MyOutputLayer:

    def __init__(self, output_size):
        self.layer_size = output_size
        self.prev_layer = None
        self.weights_and_biases_initialized = False
        self.weights = None
        self.biases = None
        self.inputs_to_layer = None
        self.current_prediction_batch = None
        self.all_pre_softmax_values = None
        self.partials_wrt_weights = None
        self.partials_wrt_biases = None

    @staticmethod
    def apply_output_activation(whole_pre_output):  # uses softmax activation function
        softmax_probabilities = []
        for layer in whole_pre_output:
            e_raised_values = np.exp(layer - np.max(layer))  # clipped to prevent overflow, now all between 0 and 1
            softmax_probabilities.append(np.divide(e_raised_values, np.sum(e_raised_values)))
        return np.array(softmax_probabilities)

    def make_predictions(self, previous_nodes_batch):
        if not self.weights_and_biases_initialized:
            self.weights = np.random.rand(self.layer_size, self.prev_layer.layer_size)
            self.weights = (self.weights - 0.5) * 2
            self.biases = np.random.rand(self.layer_size)
            self.biases = (self.biases - 0.5) * 2
            self.weights_and_biases_initialized = True
        
        self.inputs_to_layer = previous_nodes_batch
        self.current_prediction_batch = \
            np.transpose(np.dot(self.weights, np.transpose(self.inputs_to_layer)))  # multiply weights by previous layer
        self.current_prediction_batch = np.add(self.current_prediction_batch, self.biases)  # add biases to each row
        self.all_pre_softmax_values = self.current_prediction_batch  # used during backpropagation
        self.current_prediction_batch = self.apply_output_activation(self.current_prediction_batch)

        return self.current_prediction_batch

    def get_back_prop_derivatives(self, d_loss_wrt_predictions):
        
        # Get partial nodes / partial pre softmax
        partial_nodes_partial_softmax = np.empty((0, self.layer_size))
        for layer_from_an_input_in_batch in self.all_pre_softmax_values:
            current_layer_partials = []
            for node_pre_softmax in layer_from_an_input_in_batch:
                current_layer_partials.append(np.divide(np.multiply(np.exp(node_pre_softmax),
                                                                    np.add(np.sum(np.exp(layer_from_an_input_in_batch)),
                                                                           -np.exp(node_pre_softmax))),
                                                        np.power(np.sum(np.exp(layer_from_an_input_in_batch)), 2)))
            partial_nodes_partial_softmax = np.append(partial_nodes_partial_softmax, [current_layer_partials], axis=0)
        partial_nodes_partial_softmax = np.sum(partial_nodes_partial_softmax, axis=0)  # Get total partial by node

        # Get partial of loss wrt weights into output nodes
        self.partials_wrt_weights = np.repeat(np.array([np.sum(self.inputs_to_layer, axis=0)]), self.layer_size, axis=0)
        for index, node in enumerate(self.partials_wrt_weights):
            self.partials_wrt_weights[index] = np.multiply(node, np.multiply(d_loss_wrt_predictions[index],
                                                                             partial_nodes_partial_softmax[index]))

        # Get partial of loss wrt bias into output nodes
        self.partials_wrt_biases = np.multiply(d_loss_wrt_predictions, partial_nodes_partial_softmax)

        # Return partials wrt previous layer nodes
        return np.dot(np.multiply(d_loss_wrt_predictions, partial_nodes_partial_softmax), self.weights)

    def apply_back_propagation(self, total_loss, learning_rate):
        self.weights = np.add(self.weights, np.multiply(self.partials_wrt_weights,
                                                        np.multiply(total_loss, learning_rate)))
        self.biases = np.add(self.biases, np.multiply(self.partials_wrt_biases, np.multiply(total_loss, learning_rate)))
