# -*- coding: utf-8 -*-
""" Base class for building One-Class AutoEncoder
"""
import torch

import numpy as np

from torch import nn

from pyod.models.base import BaseDetector
from pyod.models.auto_encoder_torch import PyODDataset

from ..utils.tools_neural_networks import activation_with_str


class BaseAutoencoder(nn.Module, BaseDetector):
    def __init__(self,
                 input_dim: int,
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
                 verbose: int = 1):
        # invoke super class constructor
        nn.Module.__init__(self)
        BaseDetector.__init__(self, contamination=contamination)

        # store model parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.batchnorm = batchnorm
        self.pyod_preprocessing = pyod_preprocessing
        self.activation = activation_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.weight_decay = weigh_decay
        self.device = device
        self.verbose = verbose

        # create default loss functions
        #  > pyod
        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss()

        # define encoder and decoder dimensions
        self.encoder_layers = [input_dim] + hidden_dims
        self.decoder_layers = self.encoder_layers.copy()
        self.decoder_layers.reverse()

        self._build_encoder()
        self._build_decoder()

        self.fitted = False
        self.decision_scores_ = None

        self.best_loss = (0.0, 0) # (loss, epoch)
        self.best_ap =(0.0, 0) # (ap, epoch)
        self.best_auc =(0.0, 0) # (auc, epoch)

        # todo: put the print in a logger
        # print(self)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def fit(self, X, y=None):
        # the BaseAutoencoder uses Adam optimizer
        # todo: let the user chose the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # standardization of the input if required
        if self.pyod_preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            train_set = PyODDataset(X=X)
        
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        for epoch in range(self.epochs):
            overall_loss = []
            
            for batch, batch_idx in train_loader:
                batch = batch.to(self.device).float()

                outputs = self(batch)
                reconstruction_loss = self.loss_fn(batch, outputs)

                self.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()

                overall_loss.append(reconstruction_loss.item())
            
            if not self.fitted:
                self.fitted = True
                    
            if self.verbose == 1:
                print('epoch {epoch}: training loss {train_loss} '.format(
                    epoch=epoch, train_loss=np.mean(overall_loss)))

        self.fitted = True

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        if not self.fitted:
            raise ValueError("The model is not fitted.")
        
        # standardization of the input if required
        if self.pyod_preprocessing:
            data_set = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            data_set = PyODDataset(X=X)
        
        data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.eval()

        outlier_scores = np.zeros([X.shape[0], ])

        with torch.no_grad():
            for batch, batch_idx in data_loader:
                data = batch.to(self.device).float()

                euclidean_sq = np.square(self(data).cpu().numpy() - batch.numpy())
                distances = np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
                outlier_scores[batch_idx] = distances

        return outlier_scores

    def _build_layer(self,
                     i: int,
                     model: nn.Module,
                     input_dim: int,
                     output_dim: int,
                     dropout: bool = True):

        model.add_module('linear_{}'.format(i), nn.Linear(input_dim, output_dim))
        model.add_module(
            self.activation + '_' + str(i),
            activation_with_str(self.activation)
        )
        if self.batchnorm:
            model.add_module('batchnorm_{}'.format(i), nn.BatchNorm1d(output_dim))

        # if dropout:

        if self.dropout_rate > 0.0 and dropout:
            model.add_module('dropout_{}'.format(i), nn.Dropout(p=self.dropout_rate))

    def _build_encoder(self):
        # we use nn.Sequential for the encoder
        self.encoder = nn.Sequential()

        # each layer is built from the shape of self.hidden_dim
        for i, i_dim in enumerate(self.encoder_layers[:-1]):
            self._build_layer(
                i,
                self.encoder,
                i_dim,
                self.encoder_layers[i + 1]
            )

    def _build_decoder(self):
        # we use nn.Sequential for the decoder
        self.decoder = nn.Sequential()

        # each layer is built from the shape of self.hidden_dim
        for i, i_dim in enumerate(self.decoder_layers[:-1]):
            self._build_layer(
                i,
                self.decoder,
                i_dim,
                self.decoder_layers[i + 1],
                dropout=True if i < len(self.decoder_layers) - 2 else False
            )
