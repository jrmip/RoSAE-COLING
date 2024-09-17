# -*- coding: utf-8 -*-
""" Base class for building One-Class AutoEncoder
"""
import click
import torch
import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

from rosae.models.ensemble import Ensemble
from rosae.models.lne_autoencoder import LNEAutoencoder
from rosae.data.rosae_dataset import RoSAEDataset, AVAILABLE_DATASETS

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

def latent_study(X, Xx, Yy,
                 emb_dim,
                 results,
                 embedding,
                 nu,
                 generation,
                 inlier,
                 run):
    ensemble = Ensemble ([
        LNEAutoencoder(
            emb_dim,
            n_neighbors=50,
            dropout_rate=0.2,
            verbose=0,
            hidden_dims=[128, 64],
            batch_size=1000,
            epochs=100,
            latent_dim=lat,
            robust_loss=True,
            lambda_1=.1,
            lambda_2=.1,
            lambda_3=.2,
            device=device
        ).to(device=device) for lat in range(2, 66, 2)
    ], n_jobs=-1)

    ensemble.fit(X)

    for base in tqdm(ensemble.base_detectors):
        pred = base.decision_function(Xx)

        results.loc[len(results.index)] = [
            'lleae',
            roc_auc_score(Yy, pred),
            average_precision_score(Yy, pred),
            embedding, nu, generation, run, inlier, 1, 50,
            base.lambda_1, base.lambda_2, base.lambda_3,
            str(base.hidden_dims), base.latent_dim
        ]

def hidden_layer_study(X, Xx, Yy,
                       emb_dim,
                       results,
                       embedding,
                       nu,
                       generation,
                       inlier,
                       run):
    ensemble = Ensemble ([
        LNEAutoencoder(
            emb_dim,
            n_neighbors=50,
            dropout_rate=0.2,
            verbose=0,
            hidden_dims=hiddens,
            batch_size=1000,
            epochs=100,
            latent_dim=hiddens[-1],
            robust_loss=True,
            lambda_1=.1,
            lambda_2=.1,
            lambda_3=.2,
            device=device
        ).to(device=device) for hiddens in hdn_lyr
    ], n_jobs=-1)

    ensemble.fit(X)

    for base in tqdm(ensemble.base_detectors):
        pred = base.decision_function(Xx)

        results.loc[len(results.index)] = [
            'lleae',
            roc_auc_score(Yy, pred),
            average_precision_score(Yy, pred),
            embedding, nu, generation, run, inlier, 1, 50,
            base.lambda_1, base.lambda_2, base.lambda_3,
            str(base.hidden_dims), base.latent_dim
        ]

def lambda_study(X, Xx, Yy,
                   emb_dim,
                   results,
                   embedding,
                   nu,
                   generation,
                   inlier,
                   run):
    ensemble = Ensemble ([
        LNEAutoencoder(
            emb_dim,
            n_neighbors=50,
            dropout_rate=0.2,
            verbose=0,
            hidden_dims=[128, 64],
            batch_size=1000,
            epochs=100,
            latent_dim=64,
            robust_loss=True,
            lambda_1=lbd1,
            lambda_2=lbd2,
            lambda_3=lbd3,
            device=device
        ).to(device=device) for lbd1 in np.arange(.0, 1.1, .1) for lbd2 in np.arange(.0, 1.1, .1) for lbd3 in np.arange(.0, 1.1, .1)
    ], n_jobs=-1)

    ensemble.fit(X)

    for base in tqdm(ensemble.base_detectors):
        pred = base.decision_function(Xx)

        results.loc[len(results.index)] = [
            'lleae',
            roc_auc_score(Yy, pred),
            average_precision_score(Yy, pred),
            embedding, nu, generation, run, inlier, 1, 50,
            base.lambda_1, base.lambda_2, base.lambda_3,
            '[128, 64]', base.latent_dim
        ]

def ensemble_study(X, Xx, Yy,
                   emb_dim,
                   results,
                   embedding,
                   nu,
                   generation,
                   inlier,
                   run):
    ensemble = Ensemble ([
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
            lambda_1=.1,
            lambda_2=.1,
            lambda_3=.2,
            device=device
        ).to(device=device) for n in range(10, 101)
    ], n_jobs=-1)

    ensemble.fit(X)

    for base in ensemble.base_detectors:
        pred = base.decision_function(Xx)

        results.loc[len(results.index)] = [
            'lleae',
            roc_auc_score(Yy, pred),
            average_precision_score(Yy, pred),
            embedding, nu, generation, run, inlier, 1, base.n_neighbors,
            .1, .1, .2, '[128, 64]', base.latent_dim
        ]

    for n in range(2, 91):
        pred = ensemble.decision_function(Xx, n_detectors=n, X_tr=X)

        results.loc[len(results.index)] = [
            'ens',
            roc_auc_score(Yy, pred),
            average_precision_score(Yy, pred),
            embedding, nu, generation, run, inlier, n, -1,
            .1, .1, .2, '[128, 64]', 64
        ]


@click.command()
@click.option('--corpus', help='Corpus should be in {}'.format(str(AVAILABLE_DATASETS)))
@click.option('--generation', default='independent')
@click.option('--embedding', default='distill-roberta')
@click.option('--runs', default=10)
@click.option('--cache', default='.tmp')
@click.option('--nu', default=.1)
@click.option('--name', default="exp")
@click.option('--study', default='ensemble')
def run(corpus, generation, embedding, runs, cache, nu, name, study):
    device = 'cpu'
    results = pd.DataFrame(columns=[
        'model', 'auc', 'ap', 'embedding', 'nu', 'generation', 'run', 'inlier',
        'ndetector', 'n_neighbors', 'lambda1', 'lambda2', 'lambda3', 'hidden', 'latent'
    ])

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
            emb_dim=768 if 'roberta' in embedding else 300

            ds.tac(i)
            test.tac(i)

            X = torch.stack(list(ds.data_split[:,0])).numpy()

            Xx = torch.stack(list(test.data_split[:,0])).numpy()
            Yy = [0 if j == i else 1 for j in test.data_split[:,1]]

            if study == 'ensemble':
                ensemble_study(X, Xx, Yy, emb_dim, results, embedding, nu, generation, i, run)
            elif study == 'lambda':
                lambda_study(X, Xx, Yy, emb_dim, results, embedding, nu, generation, i, run)
            elif study == 'hidden':
                hidden_layer_study(X, Xx, Yy, emb_dim, results, embedding, nu, generation, i, run)
            elif study == 'latent':
                latent_study(X, Xx, Yy, emb_dim, results, embedding, nu, generation, i, run)


        results.to_pickle(Path(LOCAL_PATH / cache / 'results' / '{}.pickle'.format(name)))

if __name__ == '__main__':
    run()
