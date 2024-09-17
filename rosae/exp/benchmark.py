# -*- coding: utf-8 -*-
""" Base class for building One-Class AutoEncoder
"""
import click
import torch
import random
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

from rosae.models.ensemble import Ensemble
from rosae.models.lne_autoencoder import LNEAutoencoder
from rosae.models.base_autoencoder import BaseAutoencoder
from rosae.models.random_autoencoder import RandomAutoencoder
from rosae.data.rosae_dataset import RoSAEDataset, AVAILABLE_DATASETS
from rosae.models.rsr_autoencoder import RobustSubspaceRecoveryAutoencoder

from tqdm import tqdm
from pathlib import Path

import warnings

warnings.filterwarnings('ignore')

device = "cpu"
corpus = "imdb"

lr = [.001, .005, .01, .05, .1, .15, .2, .25, .3]
drpout = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
# wght_dc = [1.0, 2.0, 3.0, 4.0, 5.0]
hdn_lyr = [[64, 32], [128, 64, 32], [128, 64], [256, 128, 64, 32], [512, 256, 128, 64, 32]]
rdm_co = [.1, .2, .3, .4, .5]

LOCAL_PATH = Path(__file__).resolve().parents[1]

def _init_models(device, emb_dim):
    return {
        'base': BaseAutoencoder(
                emb_dim,
                dropout_rate=0.2,
                learning_rate=1e-3,
                batchnorm=True,
                hidden_dims=[128, 64],
                epochs=100,
                verbose=0,
                batch_size=1000,
                device=device
            ).to(device=device),
        'lleae': LNEAutoencoder(
                emb_dim,
                n_neighbors=50,
                dropout_rate=0.2,
                verbose=0,
                hidden_dims=[128, 64],
                batch_size=1000,
                epochs=100,
                latent_dim=64,
                robust_loss=True,
                lambda_3=.2,
                lambda_1=.1,
                lambda_2=.1,
                device=device
            ).to(device=device),
        'lleae2': LNEAutoencoder(
                emb_dim,
                n_neighbors=100,
                dropout_rate=0.2,
                verbose=0,
                hidden_dims=[128, 64],
                batch_size=1000,
                epochs=100,
                locality="inner",
                latent_dim=64,
                robust_loss=False,
                device=device
            ).to(device=device),
        'rae': RandomAutoencoder(
                emb_dim,
                dropout_rate=0.2,
                learning_rate=1e-3,
                hidden_dims=[128, 64],
                epochs=100,
                verbose=0,
                device=device,
                batch_size=1000,
                connection_prune_ratio=0.5
            ).to(device=device),
        'rsrae':RobustSubspaceRecoveryAutoencoder(
                emb_dim,
                16,
                dropout_rate=0.2,
                learning_rate=1e-3,
                hidden_dims=[128, 64],
                epochs=100,
                verbose=0,
                device=device,
                batch_size=1000,
            ).to(device=device),
        'ens': Ensemble ([
                LNEAutoencoder(
                    emb_dim,
                    n_neighbors=n,
                    dropout_rate=0.2,
                    verbose=0,
                    hidden_dims=[128, 64],
                    batch_size=1000,
                    epochs=100,
                    latent_dim=64,
                    robust_loss=True,
                    lambda_3=.2,
                    lambda_1=.1,
                    lambda_2=.1,
                    device=device
                ).to(device=device) for n in range(5, 105, 5)#np.random.randint(5, 100, 20)
            ], n_jobs=-1)
    }

@click.command()
@click.option('--corpus', help='Corpus should be in {}'.format(str(AVAILABLE_DATASETS)))
@click.option('--generation', default='independent')
@click.option('--embedding', default='distill-roberta')
@click.option('--runs', default=10)
@click.option('--cache', default='.tmp')
@click.option('--nu', default=.1)
@click.option('--name', default="exp")
def run(corpus, generation, embedding, runs, cache, nu, name):
    device = 'cpu'
    results = pd.DataFrame(
        columns=['model', 'auc', 'ap', 'embedding', 'nu', 'generation', 'run', 'inlier']
    )

    Path(LOCAL_PATH / cache/ 'results').mkdir(parents=True, exist_ok=True)

    for run in tqdm(range(runs)):
        ds = RoSAEDataset(
            corpus,
            cache_folder=cache,
            generation=generation,
            split_size=1000,
            embedding=embedding,
            encode_device=device,
            nu=nu,
            min_size=100
        )
        try:
            test = RoSAEDataset(
                corpus,
                cache_folder=cache,
                generation=generation,
                split_size=1000,
                embedding=embedding,
                encode_device=device,
                split='test' if corpus != 'sst2' else 'validation',
                nu=nu,
                min_size=100
            )
        except:
            test = RoSAEDataset(
                corpus,
                cache_folder=cache,
                generation=generation,
                split_size=1000,
                embedding=embedding,
                encode_device=device,
                split='train',
                nu=nu,
                min_size=100
            )

        for i in random.sample(ds.inliers, min(40, len(ds.inliers))):
            models = _init_models(
                device=device,
                emb_dim=768 if 'roberta' in embedding else 300
            )

            ds.tac(i)
            test.tac(i)

            X = torch.stack(list(ds.data_split[:,0])).numpy()

            Xx = torch.stack(list(test.data_split[:,0])).numpy()
            Yy = [0 if j == i else 1 for j in test.data_split[:,1]]

            for clf in models:
                models[clf].fit(X)

                results.loc[len(results.index)] = [
                    clf,
                    roc_auc_score(Yy, models[clf].decision_function(Xx)),
                    average_precision_score(Yy, models[clf].decision_function(Xx)),
                    embedding, nu, generation, run, i
                ]

        results.to_pickle(Path(LOCAL_PATH / cache / 'results' / '{}.pickle'.format(name)))

if __name__ == '__main__':
    run()
