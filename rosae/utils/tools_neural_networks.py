from torch import nn


POSSIBLE_ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}


def activation_with_str(activation: str) -> nn.Module:
    """

    Parameters
    ----------
    activation

    Returns
    -------

    """
    if activation.lower() not in POSSIBLE_ACTIVATION_FUNCTIONS.keys():
        raise ValueError(activation, "is not a compatible activation function")

    return POSSIBLE_ACTIVATION_FUNCTIONS[activation]
