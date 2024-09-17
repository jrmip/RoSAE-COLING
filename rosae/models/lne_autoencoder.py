import torch
import torch.nn as nn

import numpy as np

from rosae.models.base_autoencoder import BaseAutoencoder

from pyod.models.auto_encoder_torch import PyODDataset

class LELayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_neighbors: int = 5,
                 device:str = "cpu"):
        super(LELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neighbors = n_neighbors
        self.device = device

        # Initialize weight matrix W and bias
        self.A = nn.Parameter(
            torch.randn((input_dim, output_dim)),
            requires_grad=True
        )

    def forward(self, x, indexes_neighbors=None):
        if indexes_neighbors is None: #inner locality
            # compute pairwise distances
            pairwise_distances = torch.cdist(x, x)
            # remove self-distance from pairwise matrix
            pairwise_distances.fill_diagonal_(float("Inf"))
            # find nearest neighbors
            _, indices = torch.topk(
                pairwise_distances,
                min(self.n_neighbors, len(x)), # handle batch
                dim=1,
                largest=False
            )
        else: # outer locality
            indices = indexes_neighbors

        xW = torch.matmul(x, self.A)

        local_weights = torch.zeros(x.shape[0], self.A.shape[1], device=self.device)

        for i, ngbrs in enumerate(indices):
            local_weights[i] = torch.sum(xW[ngbrs], dim=0)

        return local_weights

class LELoss(nn.Module):
    def __init__(self,
                 lambda1: float,
                 lambda2: float,
                 lambda3: float,
                 intrinsic_size: int,
                 if_robust: bool = True,
                 norm_type: str = "MSE",
                 loss_norm_type: str = "MSE",
                 all_alt: bool = False,
                 device: str = "cpu",
                 n_neighbors:int = 10):
        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.intrinsic_size = intrinsic_size

        self.all_alt = all_alt
        self.if_robust = if_robust
        self.norm_type = norm_type
        self.loss_norm_type = loss_norm_type
        self.device = device
        self.n_neighbors = n_neighbors

    def forward(self, x, encoded, latent, decoded, rsrA):
        loss = self._reconstruction_error(x, decoded)

        if self.if_robust and not self.all_alt:
            loss += self.lambda1 * self._pca_error(encoded, latent, rsrA)
            loss += self.lambda2 * self._proj_error(rsrA)
            loss += .1 * self._local_error(encoded, latent, rsrA)

        return loss

    def _reconstruction_error(self, x, decoded):
        x = x.view(x.shape[0], -1)
        decoded = decoded.view(decoded.shape[0], -1)

        return self._norm(x, decoded, self.loss_norm_type)
    
    def _local_error(self, x, latent, emb):
        # compute pairwise distances
        pairwise_distances = torch.cdist(latent, latent)
        # remove self-distance from pairwise matrix
        pairwise_distances.fill_diagonal_(float("Inf"))
        # find nearest neighbors
        _, indices = torch.topk(
            pairwise_distances,
            min(self.n_neighbors, len(latent)), # handle batch
            dim=1,
            largest=False
        )

        local_weights = torch.zeros(latent.shape[0], latent.shape[1], device=self.device)

        for i, ngbrs in enumerate(indices):
            local_weights[i] = torch.sum(latent[ngbrs], dim=0)
        
        z = torch.matmul(latent, emb.T)
        
        return self._norm(x, z, self.norm_type)

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

class LNEAutoencoder(BaseAutoencoder):
    def __init__(self,
                 input_dim,
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
                 n_neighbors: int = 5,
                 locality: str = "inner",
                 latent_dim: int = 16,
                 robust_loss: bool = False,
                 lambda_1: float = .1,
                 lambda_2: float = .1,
                 lambda_3: float = .1,
                 randomly_connected:bool = True,
                 connection_prune_ratio:float = .2):
        self.latent_dim = latent_dim

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
        self.n_neighbors = n_neighbors
        self.locality = locality
        self.robust_loss = robust_loss

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.randomly_connected = randomly_connected
        self.connection_prune_ratio = connection_prune_ratio

        self.local_embedding = LELayer(
            hidden_dims[-1], self.latent_dim,
            n_neighbors=self.n_neighbors,
            device=self.device
        )

        self.loss_fn = LELoss(
            self.lambda_1, self.lambda_2, self.lambda_3,
            latent_dim,
            if_robust=robust_loss,
            device=self.device
        )
    
    def forward(self, x, indexes_neighbors=None):
        encoded = self.encoder(x)
        latent = self.local_embedding(encoded, indexes_neighbors)
        decoded = self.decoder(latent)
        
        return encoded, latent, decoded, self.local_embedding.A

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

                indices = None

                if self.locality == "outer":
                    # compute pairwise distances
                    pairwise_distances = torch.cdist(batch, batch)
                    # remove self-distance from pairwise matrix
                    pairwise_distances.fill_diagonal_(float("Inf"))
                    # find nearest neighbors
                    _, indices = torch.topk(
                        pairwise_distances,
                        min(self.n_neighbors, len(batch)), # handle batch
                        dim=1,
                        largest=False
                    )

                encoded, latent, decoded, emb = self(batch, indices)
                # if not self.robust_loss:
                #     reconstruction_loss = self.loss_fn(batch, decoded)
                # else:
                reconstruction_loss = self.loss_fn(
                    batch,
                    encoded,
                    latent,
                    decoded,
                    emb
                )

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
        outlier_scores = np.zeros([X.shape[0], ])
    
        with torch.no_grad():
            for batch, batch_idx in data_loader:
                data = batch.to(self.device).float()
                euclidean_sq = np.square(self(data)[2].cpu().numpy() - batch.numpy())
                distances = np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
                outlier_scores[batch_idx] = distances

        return outlier_scores
    
    def _build_decoder(self):
        layers = [self.latent_dim] + self.decoder_layers[1:]

        # we use nn.Sequential for the decoder
        self.decoder = nn.Sequential()

        # each layer is built from the shape of self.hidden_dim
        for i, i_dim in enumerate(layers[:-1]):
            self._build_layer(
                i,
                self.decoder,
                i_dim if i != 0 else self.latent_dim,
                self.decoder_layers[i + 1],
                dropout=True if i < len(self.decoder_layers) - 2 else False
            )
