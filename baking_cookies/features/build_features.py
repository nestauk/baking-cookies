import logging
import yaml
import ast
from pathlib import Path
from pandas import read_csv, DataFrame
from baking_cookies.features.w2v import train_w2v, document_vector
import baking_cookies

logger = logging.getLogger(__name__)


def main():
    """
    """
    config = baking_cookies.config['features']
    project_dir = baking_cookies.project_dir

    make_gtr(project_dir, config['w2v'])

def make_gtr(project_dir, w2v_config):
    # Input data
    fin = f"{project_dir}/data/processed/gtr_tokenised.csv"
    # Output Word2Vec model
    w2v_out = f"{project_dir}/models/gtr_w2v"
    # Output Document vectors
    docs_out = f"{project_dir}/data/processed/gtr_embedding.csv"

    logger.info('load gateway to research data')
    docs = (read_csv(fin,
            usecols=['processed_documents'])
            .processed_documents
            .apply(ast.literal_eval))


    logger.info('making gateway to research word embeddings')
    w2v = train_w2v(docs, **w2v_config)
    logger.info(f'saving gateway to research word embeddings to {w2v_out}')
    w2v.wv.save(w2v_out)

    logger.info('making gateway to research document vectors')
    doc_vecs = (DataFrame([document_vector(w2v, doc) for doc in docs],
                index=docs.index,
                columns=[f'dim_{i}' for i in range(w2v.vector_size)])
                )

    logger.info(f'saving gateway to research document vectors to {docs_out}')
    doc_vecs.to_csv(docs_out)


if __name__ == '__main__':

    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e
