"""
The main code for the recurrent and convolutional networks

"""
from typing import Tuple, List, Dict
import tensorflow
from tensorflow.keras import  layers, models, regularizers  # type: ignore # type: ignoregit
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    #### YOUR CODE HERE ####
    inputs = layers.Input(shape=input_shape)
    #SimpleRNN layer with 256 units and 'tanh' activation
    rnn_layer = layers.SimpleRNN(256, activation='tanh', return_sequences=True)(inputs)
    #Dense output layer with `n_outputs` neurons and a 'linear' activation function
    output_layer = layers.Dense(n_outputs, activation='linear')(rnn_layer)
    #model by specifying the inputs and outputs
    model = models.Model(inputs, output_layer)
    model.compile(optimizer='adam', loss='mse') #loss- Means Squared Error.
    fit_kwargs = {'batch_size': 32, 'epochs': 10} #running 10 epchos
    return model, fit_kwargs

def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    #### YOUR CODE HERE ####

    inputs = layers.Input(shape=input_shape)
    # This layer applies 32 filters of size 3x3 to extract spatial features from the input
    # `activation='relu'` applies the ReLU activation function to introduce non-linearity
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)#first convolutional layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) #first MaxPooling layer
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x) #64 kernels, second layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)#sencond
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x) #activation- relu
    outputs = layers.Dense(n_outputs, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    fit_kwargs = {'batch_size': 64, 'epochs': 10} #epochs already defined 
    return model, fit_kwargs


def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network wilor ham (non-spam). The network will be trained
    and tested on data from:l take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, f
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    #### YOUR CODE HERE ####
    vocab_size = len(vocabulary)
    inputs = layers.Input(shape=(None,))
    #length of the dense vector is 128
    x = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs)
    #LSTM layer configurations in a list so that I can use for loop.
    lstm_layers_config = [
        {"units": 128, "return_sequences": True},
        {"units": 128, "return_sequences": True},
        {"units": 64, "return_sequences": False}
    ]
    #LSTM layers and Dropout in a loop so reduce code repeat
    for layer_config in lstm_layers_config:
        x = layers.Bidirectional(
            layers.LSTM(layer_config["units"],
                        return_sequences=layer_config["return_sequences"],
                        kernel_regularizer=regularizers.l2(0.01))
        )(x)
        x = layers.Dropout(0.5)(x) #Dropout to prevent overfitting -0.5
    x = layers.BatchNormalization()(x) #BatchNormalization layer
    outputs = layers.Dense(n_outputs, activation='sigmoid')(x) #utput layer
    model = models.Model(inputs, outputs)
    #Model with Adam optimizer, binary crossentropy loss function, and accuracy metric
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #early stopping and fit keyword arguments
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    #keyword arguments for training the model
    fit_kwargs = {
        'batch_size': 32,
        'epochs': 30, #epochs already in the test file.
        'callbacks': [early_stopping],
        'validation_split': 0.2
    }
    
    return model, fit_kwargs
def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    #### YOUR CODE HERE ####
    vocab_size = len(vocabulary)
    inputs = layers.Input(shape=(None,))
    # This layer converts word indices into dense vector representations of size 128
    # `input_dim=vocab_size` specifies the size of the vocabulary, and `output_dim=128` is the dimensionality
    x = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs) #embedding vector
    #used relu as activation, 1D convolutional layer
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x) #global Max Pooling layer
    outputs = layers.Dense(n_outputs, activation='sigmoid')(x) #sigmoid Activation
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', #adam as optimizer
                  loss='binary_crossentropy',
                    metrics=['accuracy'])
    fit_kwargs = {'batch_size': 32,
                  'epochs': 12} #Arguments are batch size and epochs
    return model, fit_kwargs
