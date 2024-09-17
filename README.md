# A Robust Autoencoder Ensemble-Based Approach for Anomaly Detection in Text
PyTorch implementation of Robust Subspace Local Recovery Autoencoder Ensemble (RoSAE) considering randomly connected autoencoders for anomaly detection in text data.
This repository presents experimental materials that can reproduces results from the original work presented at COLING 2025 paper.

## Abstract
>*Anomaly detection (AD) is a fast growing and popular domain among established applications like vision and time series. We observe a rich literature for these applications, but anomaly detection in text is only starting to blossom. Recently, self-supervised methods with self-attention mechanism have been the most popular choice. While recent works have proposed a working ground for building and benchmarking state of the art approaches, we propose two principal contributions in this paper: contextual anomaly contamination and a novel ensemble-based approach. Our method, Textual Anomaly Contamination (TAC), allows to contaminate inlier classes with either independent or contextual anomalies. In the literature, it appears that this distinction is not performed. For finding contextual anomalies, we propose RoSAE, a Robust Subspace Local Recovery Autoencoder Ensemble. All autoencoders of the ensemble present a different latent representation through local manifold learning. Benchmark shows that our approach outperforms recent works on both independent and contextual anomalies, while being more robust.*

## Requirements
The code is compatible with `Python 3.8+` and every requirements can be found in `requirements.txt`.
All models inherit the [PyOD](https://github.com/yzhao062/pyod) `BaseDetector` and can benefits from the library tools.

## Installation
Once the repository has been cloned, make sure you are on the root folder and perform the installation procedure (using `pip` ):

```bash
pip install -e .
```

Our default PyTorch installation is based on the cpu version.
While careful attention has been performed regarding gpu compatibility, we advise to run experiments on cpu.

## Handling corpora
One of the key contribution of our work lies on the availability of numerous corpora from state of the art approaches, and furthermore.
Thus we propose to use the `Datasets` library from [Hugging Face](https://huggingface.co/docs/datasets/index), our `RoSAEDataset` that handles all pre-processing and embedding steps, and PyOD `DataLoader`.

### Available corpora
Any corpus from Hugging Face is basically compatible with our implementation but for this work we limit usage to corpora of our COLING 2025 subsmission.

| **Corpus**     | **Task**           | **Documents (trn)** | **Topics** | **Hierarchy** | **Code label**        |
|----------------|--------------------|:-------------:|:----------:|:-------------:|------------------|
| 20 Newsgroups  | Classification     |     11 000    |     20     |      Yes      | `newsgroups`     |
| DBPedia 14     | Classification     |    560 000    |     14     |      Yes      | `dbpedia_14`     |
| Reuters-21578  | Classification     |     6 500    |     90     |   Yes (our)   | `reuters`        |
| Web of Science | Classification     |    47 0000    |     134    |      Yes      | `web_of_science` |
| Enron          | Spam Detection     |     33 000    |      2     |       No      | `enron`          |
| SMS Spam       | Spam Detection     |     5 500     |      2     |       No      | `sms_spam`       |
| IMDB           | Sentiment Analysis |     25 000    |      2     |       No      | `imdb`           |
| SST2           | Sentiment Analysis |     67 000    |      2     |       No      | `sst2`           |

### Text embedding
Text embedding can be performed with several options: GloVe, FastText, RoBERTa, etc ...
While we have experimented numerous language models for text embedding, results recorded in our submission have been performed with Distill RoBERTa.

| **Model**       | **Dimension** | **Code label**     |
|-----------------|---------------|--------------------|
| FastText        | 300           | `fasttext`         |
| GloVe           | 300           | `sentence-glove`   |
| RoBERTa         | 768           | `roberta`          |
| Distill RoBERTa | 768           | `sentence-roberta` |

### Important notes
First, we highly recommend to use `sentence-roberta` for all experiments.
Also, for each recorded result on one corpus we perform `NB_RUN * NB_TOPICS`, which can take a long time to perform.
For avoiding to transform several times the same document, and for getting better run times, we transform one all the corpus and store it in the cache folder (default is `.tmp` of the `rosae/` folder).
Thus for a quick check of the results on one corpus, we advise to first run any experiment on smallest corpora.

## Running scripts
We propose two scripts for reproducing our experimental setup: `benchmark` and `ablation_study` (`rosae/exp/`).
Both comes with command line implementation with useful options.
For more advanced experiments, you can use your own instead through python imports.

### Benchmark
```bash
python3 rosae/exp/benchmark --corpus='dbpedia_14' --generation='independent' --embedding='distill-roberta' --runs=10 --cache='.tmp' --nu=0.1 --name="benchmark"
```

All results will be stored in a pandas dataframe in `rosae/.tmp/results/benchmark.pickle`.
An easy way for visualizing AUC and AP is:

```python
import pandas as pd
df = pd.read_pickle('erla/.tmp/results/benchmark.pickle')
df.groupby('model').auc.mean()
df.groupby('model').ap.mean()
```

A lot more informations can be found in the dataframe.

### Ablation study
```bash
python3 rosae/exp/ablation_study --corpus='reuters' --generation='contextual' --embedding='distill-roberta' --runs=10 --cache='.tmp' --nu=0.1 --name='ablation_ensemble' --study='ensemble'
```

Similar to `benchmark`, the ablation study script will store study of ensemble properties (neighbours number and detector number) in `rosae/.tmp/results/ablation_ensemble.pickle`.

The `--study` option can take four values:
* `ensemble` study of ensemble components and *k* hyperparameter for LNE embedding
* `lambda` propose several values association for the three hyperparameters
* `hidden` study impact of the number of hidden layers in one RLAE
* `latent` process numerous analysis on latent space from one RLAE

## Final note
Each result has been performed on cpu with a M1 Macbook Pro.
The embedding step after loading the Reuters-21578 corpus was as follows:

| **Step**                     |   **Time**  |
|------------------------------|:-----------:|
| Embedding of train documents |  3min 44sec |
| Embedding of test documents  |  1min 27sec |
| Benchmark with 10 runs       | 23min 04sec |

## License

BSD 2-Clause
