import numpy as np


from numpy import nan
from pathlib import Path
from nltk import word_tokenize


def create_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True)


def clean_text(text, lowercase=False, alnum=False):
    if text is nan:
        return nan

    t = text.lower() if lowercase else text

    t = word_tokenize(t)

    t = " ".join([word for word in t if word.isalnum()]) if alnum else " ".join(t)

    return t


def normalize(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))
