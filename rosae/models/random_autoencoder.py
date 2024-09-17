from torch import nn
import torch.nn.utils.prune as prune

from rosae.models.base_autoencoder import BaseAutoencoder
from rosae.utils.tools_neural_networks import activation_with_str

class RandomAutoencoder(BaseAutoencoder):
    def __init__(self,
                 input_dim: int = 300,
                 contamination: float = 0.1,
                 hidden_dims: list = [128, 64],
                 dropout_rate: float = 0.3,
                 batch_size: int = 32,
                 batchnorm: bool = True,
                 pyod_preprocessing: bool = False,
                 activation_fn: str = 'relu',
                 learning_rate: float = 1e-3,
                 epochs: int = 50,
                 weigh_decay: float = 0.0,
                 loss_fn=None,
                 device: str = "cpu",
                 verbose: int = 1,
                 connection_prune_ratio:float = .3):
        # for storing parameters to prune
        self.prune_parameters = []

        # invoke super class constructor
        super().__init__(input_dim,
                         contamination,
                         hidden_dims,
                         dropout_rate,
                         batch_size,
                         batchnorm,
                         pyod_preprocessing,
                         activation_fn,
                         learning_rate,
                         epochs,
                         weigh_decay,
                         loss_fn,
                         device,
                         verbose)
        
        self.connection_prune_ratio = connection_prune_ratio

        if self.connection_prune_ratio > 0.0:
            prune.global_unstructured(
                self.prune_parameters,
                pruning_method=prune.RandomUnstructured,
                amount=self.connection_prune_ratio
            )

        # todo: put the print in a logger
        # print(self)

    def _build_layer(self,
                     i: int,
                     model: nn.Module,
                     input_dim: int,
                     output_dim: int,
                     dropout: bool = True):
        layer_module = nn.Linear(input_dim, output_dim)

        model.add_module('linear_{}'.format(i), layer_module)

        self.prune_parameters.append( (layer_module, 'weight') )
        self.prune_parameters.append( (layer_module, 'bias') )

        model.add_module(
            self.activation + '_' + str(i),
            activation_with_str(self.activation)
        )
        
        if self.batchnorm:
            model.add_module('batchnorm_{}'.format(i), nn.BatchNorm1d(output_dim))

        if self.dropout_rate > 0.0 and dropout:
            model.add_module('dropout_{}'.format(i), nn.Dropout(p=self.dropout_rate))
