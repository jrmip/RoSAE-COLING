import torch
import numpy as np
import torch.nn.functional as F

from torch import nn

from .base_autoencoder import BaseAutoencoder

from pyod.models.auto_encoder_torch import PyODDataset

class RSRLayer(nn.Module):
    def __init__(self,
                 batch_size: int,
                 in_features: int,
                 intrinsic_size: int,
                 renorm: bool = False):
        super().__init__()

        self.batch_size = batch_size
        self.in_features = in_features
        self.intrinsic_size = intrinsic_size
        self.renorm = renorm

        self.A = nn.Parameter(
            torch.randn((self.in_features, intrinsic_size)),
            requires_grad=True
        )
        # self.A = nn.Parameter(nn.init.orthogonal_(
        #     torch.empty(self.intrinsic_size, self.encoder_dimension)))

    def forward(self, y):
        # we generalize the code for 1-2-3 dimension shapes
        yf = torch.flatten(y, start_dim=1)
        z = torch.matmul(yf, self.A)

        return z if not self.renorm else F.normalize(z, p=2, dim=-1)

    def extra_repr(self) -> str:
        rpr = ""
        rpr += "in_features={}, ".format(self.in_features)
        rpr += "out_features={}, ".format(self.intrinsic_size)
        rpr += "require_grad={}, ".format(self.A.requires_grad)
        rpr += "renorm={}".format(self.renorm)

        return rpr


class RSRLoss(nn.Module):
    def __init__(self,
                 lambda1: float,
                 lambda2: float,
                 intrinsic_size: int,
                 if_rsr: bool = True,
                 norm_type: str = "MSE",
                 loss_norm_type: str = "MSE",
                 all_alt: bool = False,
                 device: str = "cpu"):
        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.intrinsic_size = intrinsic_size

        self.all_alt = all_alt
        self.if_rsr = if_rsr
        self.norm_type = norm_type
        self.loss_norm_type = loss_norm_type
        self.device = device

    def forward(self, x, encoded, latent, decoded, rsrA):
        loss = self._reconstruction_error(x, decoded)

        if self.if_rsr and not self.all_alt:
            loss += self.lambda1 * self._pca_error(encoded, latent, rsrA)
            loss += self.lambda2 * self._proj_error(rsrA)

        return loss

    def _reconstruction_error(self, x, decoded):
        x = x.view(x.shape[0], -1)
        decoded = decoded.view(decoded.shape[0], -1)

        return self._norm(x, decoded, self.loss_norm_type)

    def _pca_error(self, y, z, rsrA):
        z = torch.matmul(z, torch.transpose(rsrA, 0, 1))

        return self._norm(y, z, self.norm_type)

    def _proj_error(self, z):
        z = torch.matmul(torch.transpose(z, 0, 1), z)
        z = torch.sub(z, torch.eye(self.intrinsic_size, device=self.device))
        z = torch.square(z)

        return torch.mean(z)

    @staticmethod
    def _norm(y, z, kind):
        if kind in ['MSE', 'mse', 'Frob', 'F']:
            return torch.mean(torch.square(torch.norm(torch.sub(y, z), p=2, dim=1)))
        elif kind in ['L1', 'l1']:
            return torch.mean(torch.norm(torch.sub(y, z), p=1, dim=1))
        elif kind in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
            return torch.mean(torch.norm(torch.sub(y, z), p=2, dim=1))
        else:
            raise Exception("Norm type error!")


class RobustSubspaceRecoveryAutoencoder(BaseAutoencoder):
    def __init__(self,
                 input_dim: int,
                 intrinsic_size: int,
                 contamination: float = 0.1,
                 hidden_dims: list = [64, 32],
                 dropout_rate: float = 0.3,
                 batch_size: int = 32,
                 batchnorm: bool = True,
                 pyod_preprocessing: bool = False,
                 activation_fn: str = 'relu',
                 learning_rate: float = 1e-3,
                 epochs: int = 50,
                 weigh_decay: float = 0.0,
                 loss_fn=None,
                 normalize: bool = False,
                 norm_type: str = "MSE",
                 loss_norm_type: str = "MSE",
                 if_rsr: bool = True,
                 lambda1: float = 0.1,
                 lambda2: float = 0.1,
                 all_alt: bool = False,
                 random_seed: int = 123,
                 verbose: int = 1,
                 device: str = "cpu"):
        if intrinsic_size >= hidden_dims[-1]:
            raise AttributeError("Manifold dimension is not inferior to latent")

        self.intrinsic_size = intrinsic_size

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

        self.norm_type = norm_type
        self.loss_norm_type = loss_norm_type
        self.if_rsr = if_rsr
        self.all_alt = all_alt
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate

        self.batch_size = batch_size

        self.normalize = normalize
        self.seed = random_seed
        self.verbose = verbose

        self.rsr = RSRLayer(
            self.batch_size,
            self.hidden_dims[-1],
            self.intrinsic_size,
            self.normalize
        )

        self.loss_fn = RSRLoss(
            self.lambda1,
            self.lambda2,
            self.intrinsic_size,
            self.if_rsr,
            self.norm_type,
            self.loss_norm_type,
            self.all_alt,
            device=device
        )

        self.device = device

        if self.verbose == 1:
            print(self)

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.rsr(encoded)
        decoded = self.decoder(latent)

        return encoded, latent, decoded, self.rsr.A

    def fit(self, X, y=None):
        if y is not None and X.size(0) != y.size(0):
            raise RuntimeError('The number of instance from X should be the same as Y')

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

            # for batch in dataloader:
            for batch, batch_idx in train_loader:
                batch = batch.to(self.device).float()

                encoded, latent, decoded, rsrA = self(batch)

                reconstruction_loss = self.loss_fn(
                    batch,
                    encoded,
                    latent,
                    decoded,
                    rsrA
                )

                self.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()

                overall_loss.append(reconstruction_loss.item())

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
        outlier_scores = np.zeros([X.shape[0], ])
    
        with torch.no_grad():
            for batch, batch_idx in data_loader:
                data = batch.to(self.device).float()
                euclidean_sq = np.square(self(data)[2].cpu().numpy() - batch.numpy())
                distances = np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
                outlier_scores[batch_idx] = distances

        return outlier_scores

    def _build_decoder(self):
        layers = [self.intrinsic_size] + self.decoder_layers[1:]

        # we use nn.Sequential for the decoder
        self.decoder = nn.Sequential()

        # each layer is built from the shape of self.hidden_dim
        for i, i_dim in enumerate(layers[:-1]):
            self._build_layer(
                i,
                self.decoder,
                i_dim if i != 0 else self.intrinsic_size,
                self.decoder_layers[i + 1],
                dropout=True if i < len(self.decoder_layers) - 2 else False
            )
