import torch
import numpy as np

from torch import nn

from .base_autoencoder import BaseAutoencoder


class RandomlyConnectedAutoencoder(BaseAutoencoder):
    def __init__(self,
                 input_dim: int,
                 contamination: float = 0.1,
                 hidden_dims: list = [128, 64],
                 dropout_rate: float = 0.0,
                 batch_size: int = 32,
                 batchnorm: bool = True,
                 activation_fn: str = 'relu',
                 learning_rate: float = 1e-3,
                 epochs: int = 50,
                 weigh_decay: float = 0.0,
                 loss_fn=None,
                 connection_prob: float = 0.5,
                 verbose: int = 1,
                 device: str = 'cpu'):
        super().__init__(input_dim,
                         contamination,
                         hidden_dims,
                         dropout_rate,
                         batch_size,
                         batchnorm,
                         activation_fn,
                         learning_rate,
                         epochs,
                         weigh_decay,
                         loss_fn,
                         device,
                         verbose)

        self.connection_mask = None
        self.connection_prob = connection_prob

        self._randomize_connection(connection_probability=self.connection_prob)

    def fit(self, X, y=None):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            overall_loss = []

            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size].to(self.device)

                outputs = self(batch)
                reconstruction_loss = self.loss_fn(outputs, batch)

                self._update_weight(optimizer, reconstruction_loss)

                overall_loss.append(reconstruction_loss.item())

                if self.verbose == 1:
                    print('epoch {epoch}: training loss {train_loss} '.format(
                        epoch=epoch, train_loss=np.mean(overall_loss)))

        self.fitted = True

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        return self

    def _randomize_connection(self, connection_probability=0.5):
        self.connection_mask = []
        for name, module in self.named_children():
            if not isinstance(module, nn.MSELoss):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        weight = layer.weight.data
                        mask = np.random.binomial(1, connection_probability, weight.shape)
                        self.connection_mask.append(mask)
                        weight *= torch.tensor(mask, dtype=weight.dtype)
                        layer.weight.data = weight

    def _update_weight(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()

        i = 0
        for (name, module) in self.named_children():
            if not isinstance(module, nn.MSELoss):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        layer.weight.grad.data *= torch.tensor(
                            self.connection_mask[i],
                            dtype=layer.weight.grad.data.dtype).to(self.device)

                        layer.weight.data *= torch.tensor(
                            self.connection_mask[i],
                            dtype=layer.weight.data.dtype
                        ).to(self.device)
                        i += 1
        optimizer.step()
