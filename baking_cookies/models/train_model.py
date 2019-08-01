import logging
import ast
import yaml
import joblib
from pathlib import Path
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
import baking_cookies

logger = logging.getLogger(__name__)


def make_train_model(project_dir, grid_search=False):
    """ Train model

    Args:
        project_dir (str):
        grid_search (bool, optional): If True, perform grid_search, otherwise
            fit best parameters only (see `model_config.yaml`)
    """

    target = baking_cookies.config['target']

    gtr_fin = f"{project_dir}/data/processed/gtr_tokenised.csv"
    train_id_fin = f"{project_dir}/data/processed/gtr_leadFunder_id_train.csv"
    model_fout = f"{project_dir}/models/gtr_{target}.pkl"

    if grid_search:
        grid_search_kws = baking_cookies.config['model']['grid']
    else:
        grid_search_kws = baking_cookies.config['model']['best']

    model = (
             # Load data
             read_csv(gtr_fin, converters={'processed_documents': ast.literal_eval})
             # Get train indexes
             .merge(read_csv(train_id_fin), on='id')
             # Fit model
             .pipe(lambda x: process_train_model(
                   x['processed_documents'],  # Features
                   x[target],  # Target
                   grid_search_kws,
                   ))
             )

    # Save pipeline
    joblib.dump(model, model_fout)


def _identity(x):
    return x

def process_train_model(x_train, y_train, grid_search_kws):
    """

    Args:
        x_train (pandas.DataFrame): Train features
        y_train (pandas.DataFrame): Train labels
        grid_search_kws: Grid search kwargs. Must include `param_ grid`.

    Returns:
        sklearn.pipeline.Pipeline
    """

    assert 'param_grid' in grid_search_kws, 'Must provide parameter grid'


    estimators = [
            ('tfidf', TfidfVectorizer(preprocessor=_identity, tokenizer=_identity)),
            ('logr', LogisticRegression()),
            ]


    pipe = Pipeline(estimators)

    if 'random_state' in grid_search_kws:
        random_state = grid_search_kws['random_state']
    else:
        random_state = 0

    if 'cv' in grid_search_kws:
        n_splits = grid_search_kws['cv']
    else:
        n_splits = 3

    grid_search_kws['cv'] = KFold(n_splits=n_splits,
            random_state=random_state)

    clf = GridSearchCV(pipe, **grid_search_kws)

    clf.fit(x_train, y_train)

    logger.info(clf.best_params_)
    return clf.best_estimator_


if __name__ == '__main__':
    grid_search = False  # Don't perform a grid_search (use preset parameters)

    try:
        make_train_model(baking_cookies.project_dir, grid_search)
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e
