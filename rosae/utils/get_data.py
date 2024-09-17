import yaml

import pandas as pd

from pathlib import Path
from datasets import load_dataset
from joblib import Parallel, delayed
from rosae.utils.tools import create_if_not_exists, clean_text


LOCAL_PATH = Path(__file__).resolve().parents[1]

CORPUS_TO_HUGGINGFACE = {
    "dbpedia_14": "dbpedia_14",
    "enron": "SetFit/enron_spam",
    "imdb": "imdb",
    "newsgroups": "SetFit/20_newsgroups",
    "reuters": "reuters21578",
    "sms_spam": "sms_spam",
    "sst2": "sst2",
    "web_of_science": "web_of_science", # default subset should be WOS46985
    "ag_news": "ag_news"
}


def get_corpus(
        dataset: str,
        subset: str = '',
        split: str = 'train',
        verbose: int = 0,
        n_jobs:int = -1,
        cache_folder:str = '.tmp',
        lowercase:bool = False,
        alnum:bool = False):
    # we verify if dataset is already downloaded
    dump_path = Path(LOCAL_PATH / cache_folder / 'raw' / split)
    if Path(dump_path / '{}.pickle'.format(dataset)).exists():
        # print("Raw corpus {} already retrieved".format(dataset))
        return

    # check if the data directory already exists
    create_if_not_exists(dump_path)

    # load configuration file
    with open(Path(LOCAL_PATH / '.hierarchies'), 'r') as f:
        cfg = yaml.safe_load(f)
    
    if dataset == 'reuters':
        # load and inverse label configuration
        categories = {v: k for k, v in cfg['reuters']['categories'].items()}

    corpus = CORPUS_TO_HUGGINGFACE[dataset]

    print("Retrieving dataset : {}".format(dataset))

    if subset == '':
        ds = load_dataset(corpus, cache_dir=Path(LOCAL_PATH / cache_folder))
    else:
        ds = load_dataset(corpus, subset, cache_dir=Path(LOCAL_PATH / cache_folder))

    data = pd.DataFrame(ds[split])

    X = pd.DataFrame()

    if dataset == 'reuters':
        X['label'] = data['topics']
    else:
        X['label'] = data['label']

    if 'content' in data.columns:
        X['text'] = data['content']
    elif 'input_data' in data.columns:
        X['text'] = data['input_data']
    elif 'sentence' in data.columns:
        X['text'] = data['sentence']
    elif 'sms' in data.columns:
        X['text'] = data['sms']
    else:
        X['text'] = data['text']
    
    # post processing for reuters corpus
    if dataset == 'reuters':
        # we remove empty documents
        X = X[[len(x) == 1 and x[0] in categories for x in X['label']]]

        X['label'] = X['label'].apply(lambda l: categories[l[0]])
    
    print("Starting cleaning raw text from : {}".format(dataset))

    X['text'] = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(clean_text)(t, alnum=alnum, lowercase=lowercase) for t in X['text']
    )

    X.sample(frac=1).reset_index(drop=True)

    X.to_pickle(Path(dump_path / '{}.pickle'.format(dataset)))

    print("Saving raw corpus : {}".format(dataset))
