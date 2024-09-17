import yaml
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from rosae.utils.get_data import get_corpus
from rosae.models.embedding import Embedding

LOCAL_PATH = Path(__file__).resolve().parents[1]

AVAILABLE_DATASETS = [
    "dbpedia_14",
    "enron",
    "imdb",
    "newsgroups",
    "reuters",
    "sms_spam",
    "sst2",
    "web_of_science",
    "ag_news"
]

AVAILABLE_EMBEDDINGS = [
    'glove',
    'roberta',
    'fasttext',
    'distill-roberta',
    'sentence-glove'
]

AVAILABLE_GENERATION = [
    'independent',
    'contextual'
]


class RoSAEDataset(Dataset):
    def __init__(self,
                 dataset: str,
                 embedding: str = 'sentence-glove',
                 device: str = 'cpu',
                 cache_folder: str = ".tmp",
                 nu: float = .1,
                 split_size: int = 350,
                 min_size: int = 350,
                 generation: str = 'independent',
                 split:str = "train",
                 embedding_reduction: str = 'mean',
                 standardization: bool = False,
                 encode_device: str = "cpu",
                 alnum:bool = False,
                 lowercase:bool = False,
                 n_jobs:int = -1,
                 verbose:int = 0,
                 max_document_per_label: int = 5000):
        if dataset.lower() not in AVAILABLE_DATASETS:
            raise ValueError(
                "{} is not an available dataset. Available dataset are : "
                "{}".format(dataset, str(AVAILABLE_DATASETS))
            )

        if embedding.lower() not in AVAILABLE_EMBEDDINGS:
            raise ValueError(
                "{} is not an available embedding. Available embeddings are : "
                "{}".format(embedding, str(AVAILABLE_EMBEDDINGS))
            )

        if generation.lower() not in AVAILABLE_GENERATION:
            raise ValueError(
                "{} is not an available contamination kind. "
                "Available contaminations are : "
                "{}".format(generation, str(AVAILABLE_GENERATION))
            )

        self.dataset = dataset
        self.embedding = embedding
        self.device = device
        self.cache_folder = cache_folder
        self.split = split
        self.embedding_reduction = embedding_reduction
        self.standardization = standardization
        self.encode_device = encode_device
        self.alnum = alnum
        self.lowercase = lowercase
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_document_per_label = max_document_per_label

        self.nu = nu
        self.split_size = split_size
        self.min_size = min_size
        self.generation = generation

        self._download_raw_data()

        with open(Path(LOCAL_PATH / ".hierarchies"), 'r') as f:
            self.hierarchies = yaml.safe_load(f)

        self.corpus = pd.read_pickle(
            Path(LOCAL_PATH / self.cache_folder / "raw" / split / "{}.pickle".format(dataset)))

        # ensure that only non-empty text are kept
        self.corpus = self.corpus[self.corpus.text.astype(str).str.len() > 0]

        # verify that the embedding is available in the cache repository
        emb_repos = Path(LOCAL_PATH / self.cache_folder / "processed" / split / self.dataset)

        if Path(emb_repos / "{}.pickle".format(self.embedding)).is_file():
            self.corpus = pd.read_pickle(
                Path(emb_repos / "{}.pickle".format(self.embedding)))
        else:
            if self.embedding == 'glove':
                model = Embedding(
                    pretrained="glove",
                    cache=self.cache_folder,
                    embedding_reduction=self.embedding_reduction
                )
            elif self.embedding == "fasttext":
                model = Embedding(
                    pretrained="fasttext",
                    cache=self.cache_folder,
                    embedding_reduction=self.embedding_reduction
                )
            else:
                model = Embedding(
                    pretrained=embedding,
                    cache=self.cache_folder,
                    device=self.encode_device
                )

            tqdm.pandas(desc='Applying word embedding')

            minimized = pd.DataFrame(columns=self.corpus.columns)

            for lbl in self.corpus.label.unique():
                inlrs = self.corpus[self.corpus.label == lbl]
                minimized = pd.concat([
                    minimized,
                    inlrs.sample(self.max_document_per_label) if len(inlrs) > self.max_document_per_label else inlrs
                ], ignore_index=True)
            
            self.corpus = minimized

            self.corpus['text'] = self.corpus['text'].progress_apply(
                lambda t: model.encode(
                    t, device=self.encode_device, convert_to_tensor=True
                ).cpu()
            )

            emb_repos.mkdir(parents=True, exist_ok=True)
            self.corpus.to_pickle(Path(emb_repos / "{}.pickle".format(self.embedding)))

            # free memory
            del model

        # the dataframe should have 2 columns but the order can be a problem in
        #  the tac function
        col = self.corpus.pop('label')

        # we place the label after the raw data
        self.corpus['label'] = col

        self.data_split = None
        self.inliers = []

        if self.generation == 'contextual':
            parents_dict = self.hierarchies[self.dataset]['parents']
            # only parent inliers with more than one child are considered
            for i in parents_dict:
                p_count = 0
                for p in parents_dict.values():
                    if p == parents_dict[i]:
                        if p_count == 1:
                            inlier_len = self.corpus['label'][self.corpus['label'] == i].size

                            if  inlier_len >= self.min_size:
                                self.inliers.append(i)
                            break
                        else:
                            p_count += 1
        else:
            for i in np.unique(self.corpus['label']):
                if self.corpus['label'][self.corpus['label'] == i].size >= self.min_size:
                    self.inliers.append(i)

    def __len__(self):
        return len(self.corpus) if self.data_split is None else len(self.data_split)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.data_split is None:
            sample = self.corpus.iloc[idx]
            sample = (sample['text'], sample['label'])
        else:
            sample = self.data_split[idx]
            sample = (sample[0], sample[1])

        return sample

    def tac(self, inlier: int):
        if self.dataset in self.hierarchies:
            P = np.array([
                self.hierarchies[self.dataset]['parents'][e]
                for e in self.corpus.label]
            ).reshape(len(self.corpus), 1)
        else:
            P = self.corpus['label'].copy()

        self.data_split = self._contamination(
            (inlier, self.hierarchies[self.dataset]['parents'][inlier])
            if self.dataset in self.hierarchies else (inlier, inlier),
            self.corpus.label,
            P
        )

    def _contamination(self,
                       inlier: tuple,
                       topics: np.ndarray,
                       parents: np.ndarray):
        inlier_size = min(len(topics[topics == inlier[0]]), self.split_size)

        c = int(inlier_size * self.nu)

        self.corpus['parents'] = parents

        A = self.corpus.copy().to_numpy()

        np.random.shuffle(A)

        if self.generation == 'independent':
            outliers = A[(A[:, -2] != inlier[0]) & (A[:, -1] != inlier[1])]
        elif self.generation == 'contextual':
            outliers = A[(A[:, -2] != inlier[0]) & (A[:, -1] == inlier[1])]
        else:
            raise ValueError('Generation should be either: [independent, contextual]')

        np.random.shuffle(outliers)
        inliers = A[(A[:, -2] == inlier[0])]

        outliers = outliers[:max(c, 1)]
        inliers = inliers[:inlier_size - c]

        if len(outliers) == 0:
            raise AttributeError('No outlier found for class : {}'.format(inlier[0]))

        B = np.concatenate((outliers, inliers))

        np.random.shuffle(B)

        return B

    def _download_raw_data(self):
        if self.dataset == "web_of_science":
            subset = "WOS46985"
        elif self.dataset == "reuters":
            subset = "ModApte"
        else:
            subset = ""

        get_corpus(
            self.dataset,
            subset=subset,
            split=self.split,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            cache_folder=self.cache_folder,
            lowercase=self.lowercase,
            alnum=self.alnum
        )
