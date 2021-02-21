import os
import logging
import gensim
import numpy as np
import baking_cookies

logger = logging.getLogger(__name__)


def train_w2v(docs, **kwargs):
    """Train Word2Vec model

    Thin wrapper around gensim.models.Word2Vec that checks
    reproducibility due to `workers` arguement and
    `PYTHONHASHSEED` environment variable.

    Args:
        docs (list): List of list of tokens to train model
        **kwargs: Arguments to be passed to gensim.models.Word2Vec

    Returns:
        gensim.models.Word2Vec
    """

    if 'PYTHONHASHSEED' in os.environ.keys():
        PYTHONHASHSEED = os.environ['PYTHONHASHSEED']
        msg = f'Using env var PYTHONHASHSEED={PYTHONHASHSEED}'
        logger.info(msg)
    else:
        msg = f'No env var PYTHONHASHSEED - results not reproducible'
        logger.warning(msg)

    if 'workers' in kwargs.keys() and kwargs['workers'] != 1:
        msg = f'Workers is not equal to one - results not reproducible'
        logger.warning(msg)

    w2v = gensim.models.Word2Vec(sentences=docs, **kwargs)

    return w2v


def document_vector(word2vec_model, doc):
    """Construct and return document vectors from word embeddings

    Args:
        word2vec_model (gensim.models.Word2Vec):
            Word2Vec model

        doc (list of str):
            List of tokenised documents to construct a document vector for.

    Returns:
        numpy.array

    #UTILS
    """
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.vocab]

    if len(doc) > 0:
        return np.mean(word2vec_model[doc], axis=0)
    else:
        return np.zeros(word2vec_model.trainables.layer1_size,)
