import torch
import numpy as np

from pathlib import Path

from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe, FastText
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, RobertaModel, logging

logging.set_verbosity_error()

LOCAL_PATH = Path(__file__).resolve().parents[1]


class Embedding:
    """

    Parameters
    ----------
    tokenizer : str, optional (default="basic_english")
        Kind of torch tokenizer to use

    pretrained : str, optional (default="glove")
        The word embedding model to use.
        Available are "glove", "fasttext", "distillbert" and "roberta".

    glove_kind : str, optional (default="6B")
        Specifies which glove model to use.
        Can be either "6B", "42B", "840B" or "twitter.27B".

    language : str, optional (default="en")
        The language of the tokenizer and of the word embedding
        model if the option if available.

    dim : int, optional (default=300)
        The dimension of the final representation.

    Attributes
    ----------
    embedding_model : Torch Object
        The embedding model

    tokenizer : Torch Object
        Tokenizer from torch
    """

    def __init__(self,
                 tokenizer: str = "basic_english",
                 pretrained: str = "glove",
                 glove_kind: str = "6B",
                 language: str = "en",
                 dim: int = 300,
                 cache: str = None,
                 embedding_reduction: str = "mean",
                 device: str = "cpu"):
        self.cache = cache
        self.embedding_reduction = embedding_reduction
        self.pretrained = pretrained
        self.tokenizer = tokenizer
        self.glove_kind = glove_kind
        self.language = language
        self.dim = dim
        self.device = device

        if pretrained == "glove":
            self.embedding_model = GloVe(name=glove_kind, dim=dim, cache=Path(LOCAL_PATH / cache))
        elif pretrained == "sentence-glove":
            self.embedding_model = SentenceTransformer(
                'average_word_embeddings_glove.6B.300d',
                device=self.device,
                cache_folder=Path(LOCAL_PATH / cache)
            )
        elif pretrained == "fasttext":
            self.embedding_model = FastText(language=language, cache=Path(LOCAL_PATH / cache))
        elif pretrained == "roberta":
            self.embedding_model = RobertaModel.from_pretrained(
                "roberta-base", cache_dir=Path(LOCAL_PATH / cache)).to(self.device)
        elif pretrained == "distill-roberta":
            self.embedding_model = SentenceTransformer(
                'all-distilroberta-v1',
                device=self.device,
                cache_folder=Path(LOCAL_PATH / cache)
            )

        if self.pretrained == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained(
                'roberta-base', cache_dir=Path(LOCAL_PATH / cache), device=self.device)
        else:
            self.tokenizer = get_tokenizer(tokenizer, language=language)

    def encode(self,
               data: np.ndarray,
               lower_case_backup: bool = True,
               convert_to_tensor: bool = True,
               device: str = "cpu") -> torch.Tensor:
        if len(data) == 0:
            print("Error")

        if self.embedding_reduction == "max":
            agg = torch.max
        elif self.embedding_reduction == "sum":
            agg = torch.sum
        else:
            agg = torch.mean

        if self.pretrained == "roberta":
            tokens = self.tokenizer(
                data,
                return_tensors="pt",
                #padding=True,
                truncation=True
            ).to(device)
            X = self.embedding_model(**tokens).last_hidden_state.mean(dim=1).squeeze()
        elif self.pretrained == "distill-roberta" or self.pretrained == "sentence-glove":
            X = self.embedding_model.encode(data, convert_to_tensor=convert_to_tensor)
        else:
            X = agg(self.embedding_model.get_vecs_by_tokens(
                self.tokenizer(data), lower_case_backup=lower_case_backup
            ), 0)

        return X if self.embedding_reduction == 'mean' else X[0]
