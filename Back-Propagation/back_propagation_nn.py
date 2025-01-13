"""
The main code for the back propagation.
"""
import math
from typing import List
import numpy as np
from scipy.special import expit  # Importing expit for sigmoid function


# NOTE: In the docstrings, "UDL" refers to the book Pierce (2023),
#       "Understanding Deep learning".


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation,
    including at final layer (output) of model, and there are no bias term
    parameters, only layer weights. Input, output and weight matrices follow
    denominator layout format (same as UDL).
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer (including input (first) and output (last) layers).

        Example: create a network, net, with input layer of 3 units, a first
        hidden layer with 4 hidden units, a second hidden layer with 5 hidden
        units, and an output layer with 2 units:
            net = SimpleNetwork.random(3, 4, 5, 2)

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            # Calculate epsilon for initializing weights within a specific range
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            # Initialize weights uniformly in the range [-epsilon, +epsilon]
            return np.random.uniform(-epsilon, +epsilon, size=(n_out, n_in))
        # Create pairs of consecutive layer sizes (e.g., (layer1, layer2), (layer2, layer3))
        pairs = zip(layer_units, layer_units[1:])

        # Generate weight matrices for each pair of layers
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights specify linear transformations from one layer to the next, so
        the number of layers is equal to one more than the number of layer_weights
        weight matrices.

        :param layer_weights: A list of weight matrices
        """
        #### YOUR CODE HERE ####
        self.layer_weights = layer_weights  # Store the weight matrices
        # The number of layers is the number of weight matrices + 1 (input layer)
        self.num_layers = len(layer_weights) + 1

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a (logistic) sigmoid activation function. This includes
        at the final layer of the network.

        (This network does not include bias parameters.)

        :param input_matrix: The matrix of inputs to the network, where each
        column represents an input instance for which the neural network should make a prediction.
        :return: A matrix of predictions, where each column is the predicted
        outputs—each in the value range (0, 1)—for the corresponding column in the input matrix.
        """
        #### YOUR CODE HERE ####

        result = None  # Initialize the result variable

        # Iterate over each layer's weights
        for layer_weight in self.layer_weights:
            # Compute the weighted sum (pre-activation)
            result = np.dot(layer_weight, input_matrix)
            # Apply the sigmoid activation function
            result = 1 / (1 + np.exp(-result))
            # Set the input for the next layer to be the output of the current layer
            input_matrix = result

        return result  # Return the final output after the last layer

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an instance for which the neural network
        should make a prediction.
        :return: A matrix of predictions, where each column is the predicted
        outputs - each either 0 or 1 - for the corresponding column in the input
        matrix.
        """
        #### YOUR CODE HERE ####

        # Get the continuous output predictions
        result_matrix = self.predict(input_matrix)
        # Convert outputs to binary (0 or 1) based on a threshold of 0.5
        results = np.where(result_matrix >= 0.5, 1, 0)
        return results

    def gradients(self,
                  input_matrix: np.ndarray,
                  target_output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs backpropagation to calculate the gradients (derivatives of
        the loss with respect to the model weight parameters) for each of the
        weight matrices.

        This method first performs a pass of forward propagation through the
        network, where the pre-activations (f) and activations (h) of each
        layer are stored. (NOTE: this bookkeeping could be performed in
        self.predict(), if desired.)
        This method then applies the following procedure to calculate the
        gradients.

        In the following description, × is matrix multiplication, ⊙ is
        element-wise product, and ⊤ is matrix transpose. The acronym 'w.r.t.'
        is shorhand for "with respect to".

        First, calculate the derivative of the squared loss w.r.t. model's
        final layer, K, activations, Sig[f_K], and the target output matrix, y:

            dl_df[K] = (Sig[f_K] - y)^2

        Then for each layer k in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.

        (1) Calculate the derivatives of the loss w.r.t. the weights at the
        layer, dl_dweights[layer] (i.e., the parameter gradients), using the
        derivative of the loss w.r.t. layer pre-activation, dl_df[layer], and
        the activation, h[layer].
        (UDL equation 7.22)
        NOTE: With multiple inputs, there will be one gradient per input
        instance, and these must be summed (element-wise across gradient per
        input) and the resulting summed gradient must be (element-wise) divided
        by the number of input instances. As discussed in class, the simultaneous
        outer product and sum across gradients can be achieved using numpy.matmul,
        leaving only the element-wise division by the number of input instances.
        NOTE: The gradient() method returns the list of gradients per layer,
        so you will need to store the computed gradient per layer in a List
        for return at the end. The order of the gradients should be in
        "forward" order (layer 0 first, layer 1 second, etc...).

        (2) Calculate the derivatives of the loss w.r.t. the activations,
        dl_dh[layer], from the transpose of the weights, weights[layer].⊤,
        and the derivatives of the next pre-activation, dl_df[layer].
        (the second part of the last line of UDL equation 7.24)

        (3) If the current layer is not the 0'th layer, then:
        Calculate the derivatives of the loss w.r.t. the pre-activation
        for the previous layer, dl_df[layer - 1]. This involves the derivative
        of the activation function (sigmoid), dh_df.
        (first part of the last line of UDL eq 7.24)

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an instance for which the neural network
        should make a prediction.
        :param target_output_matrix: A matrix of expected outputs, where each column
        is the expected outputs - each either 0 or 1 - for the corresponding column
        in the input matrix.
        :return: List of the gradient matrices, 1 for each weight matrix, in same
        order as the weight matrix layers of the network, from input to output.
        """
        #### YOUR CODE HERE ####

        # Forward pass
        # Initializing sigmoid with expit
        def sigmoid(x):
            return expit(x)

        # Derivative of the sigmoid function
        def sigmoid_derivative(x):
            s = expit(x)
            return s * (1 - s)

        # Forward pass: computing activations and intermediate weighted sums
        activations_mat = [input_matrix]  # List to store activations at each layer
        weight_sums = []  # Store weighted sums (pre-activation values)

        # Iterate over each layer's weights to compute forward propagation
        for weight_matrix in self.layer_weights:
            # Compute the weighted sum (z) for the current layer
            z = np.dot(weight_matrix, input_matrix)
            weight_sums.append(z)  # Store the pre-activation value
            # Apply the sigmoid activation function
            #activation = sigmoid(z)
            activations_mat.append(sigmoid(z))
            input_matrix = sigmoid(z)

        # Initialize delta as the difference between actual and target outputs
        delta = activations_mat[-1] - target_output_matrix  # Error at the output layer
        gradients = []  # List to hold gradients for each weight matrix
        ip_counts = input_matrix.shape[0]  # Number of training examples (columns in input_matrix)

        # Backpropagation: computing gradients for each layer starting from the last layer
        for layers in range(len(self.layer_weights) - 1, -1, -1):
            # Compute the derivative of the activation function
            sigmoid_grad = sigmoid_derivative(weight_sums[layers])
            # Element-wise multiplication of delta and derivative of activation function
            delta = delta * sigmoid_grad
            # Compute gradient for the current layer's weights
            dl_df = np.dot(delta, activations_mat[layers].T) / ip_counts
            gradients.append(dl_df)  # Store the gradient
            # Compute delta for the previous layer (if not the first layer)
            delta = np.dot(self.layer_weights[layers].T, delta)

        # Reverse the gradients list to match the order of layers from input to output
        return gradients[::-1]

    def train(self,
              input_matrix: np.ndarray,
              target_output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an input instance for which the neural
        network should make a prediction
        :param target_output_matrix: A matrix of expected outputs, where each
        column is the expected outputs - each either 0 or 1 - for the corresponding row in
        the input instance in the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        #### YOUR CODE HERE ####

        for _ in range(iterations):
            #Calculate gradients via backpropagation
            gradients = self.gradients(input_matrix, target_output_matrix)
            #Update each weight matrix by subtracting (learning rate * gradient)
            self.layer_weights = list(self.layer_weights)  # Ensure weights are mutable
            for i, _ in enumerate(self.layer_weights):
                # Update weights for the current layer
                self.layer_weights[i] -= learning_rate * gradients[i]
