"""
The main code for the feedforward networks

"""
from typing import Tuple, Dict

import tensorflow
from tensorflow.keras.layers import BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

#Helper function, this is to optimise my code.
def build_model(n_inputs: int,
                n_outputs: int,
                hidden_layers: list,
                output_activation: str = None,
                loss: str = None,
                optimizer: str = 'adam',
                metrics: list = None,
                use_dropout: bool = False,
                dropout_rate: float = 0.4,
                use_batch_norm: bool = False) -> tensorflow.keras.models.Model:
    """ 
    Builds and compiles a neural network model based on the provided parameters. 
    I have submitted the code before(Pushed on gitHub)
    In the sprit of optimising the program I created a helper funtion. 
    This function must reduce the redundancy
    Astha Prassd and I discussed together for this apporach.
    """
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))

    #Add hidden layers
    for layer_params in hidden_layers:
        units = layer_params.get('units')
        activation = layer_params.get('activation')
        model.add(Dense(units, activation=activation))
        if use_batch_norm:
            model.add(BatchNormalization())
        if use_dropout:
            model.add(Dropout(dropout_rate))

    #Add output layer
    model.add(Dense(n_outputs, activation=output_activation))

    #Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.
    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    #### YOUR CODE HERE ####
       #Deep neural network parameters
    deep_hidden_layers = [
        {'units': 64, 'activation': 'relu'},
        {'units': 32, 'activation': 'relu'},
        {'units': 16, 'activation': 'relu'}
    ]
    #ctrated a deep network with 4 hidden layer
    deep_model = build_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=deep_hidden_layers,
        loss='mse',
        metrics=['mae']
    )

    #Wide neural network parameters

    wide_hidden_layers = [
        {'units': 350, 'activation': 'relu'},
    ]
    #single hidden layer
    wide_model = build_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=wide_hidden_layers,
        loss='mse',
        metrics=['mae']
    )

    return deep_model, wide_model


def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    #### YOUR CODE HERE ####

    #Common parameters - 2 layers
    hidden_layers_relu = [
        {'units': 128, 'activation': 'relu'},
        {'units': 64, 'activation': 'relu'},
    ]
    hidden_layers_tanh = [
        {'units': 128, 'activation': 'tanh'},
        {'units': 64, 'activation': 'tanh'},
    ]

    #ReLU model
    relu_model = build_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=hidden_layers_relu,
        output_activation='sigmoid',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #tanh model
    tanh_model = build_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=hidden_layers_tanh,
        output_activation='sigmoid',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return relu_model, tanh_model

def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    #### YOUR CODE HERE ####
    #Common hidden layers - 2layers
    hidden_layers = [
        {'units': 128, 'activation': 'relu'},
        {'units': 64, 'activation': 'relu'},
    ]

    #Model with dropout
    dropout_model = build_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=hidden_layers,
        output_activation='softmax',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        use_dropout=True,
        dropout_rate=0.5
    )

    #Model without dropout
    nodropout_model = build_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=hidden_layers,
        output_activation='softmax',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        use_dropout=False
    )

    return dropout_model, nodropout_model



def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    #### YOUR CODE HERE ####
    #Function to create the model architecture
    hidden_layers = [
        {'units': 128, 'activation': 'relu'},
        {'units': 64, 'activation': 'relu'},
        {'units': 32, 'activation': 'relu'},
        
    ]

    #Create the model
    def create_model():
        return build_model(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_layers=hidden_layers,
            output_activation='sigmoid',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            use_batch_norm=True  #Used batch normalization to improve my model
        )

    #Early stopping model
    early_stopping_model = create_model()
    #No early stopping model
    no_early_stopping_model = create_model()

    #Fit parameters
    early_stopping_params = {
        'callbacks': [EarlyStopping(monitor='val_loss', patience=24)],
        'epochs': 100,
        'validation_split': 0.2
    }

    no_early_stopping_params = {
        'validation_split': 0.2,
        'epochs': 100
    }

    return (early_stopping_model, early_stopping_params,
            no_early_stopping_model, no_early_stopping_params)
