import logging
import yaml
import ast
from pathlib import Path
from pandas import read_csv
from sklearn.model_selection import train_test_split
import baking_cookies

logger = logging.getLogger(__name__)


def make_train_test_split(project_dir):
    """ Performs test-train split """

    # Load config
    config = baking_cookies.config
    target = config['target']

    # Input datasets
    target_fin = f"{project_dir}/data/processed/gtr_tokenised.csv"
    # Training set output
    train_fout = f"{project_dir}/data/processed/gtr_{target}_id_train.csv"
    # Test set output
    test_fout = f"{project_dir}/data/processed/gtr_{target}_id_test.csv"

    logger.info('Loading gateway to research data')
    X_train, X_test, y_train, y_test = (
            read_csv(target_fin, index_col=0)
            .pipe(process_train_test_split, target, config['split'])
            )

    X_train.to_csv(train_fout, columns=['id'], index=False)
    msg = f'Saved training set id\'s to {train_fout}'
    logger.info(msg)

    X_test.to_csv(test_fout, columns=['id'], index=False)
    msg = f'Saved test set id\'s to {test_fout}'
    logger.info(msg)


def process_train_test_split(Xy, target, split_kwargs):
    """ Split input data into random train and test subsets

    Args:
        Xy (pandas.DataFrame): Data
        target (str): Name of target variable
        split_kwargs (dict): Keyword arguments to `train_test_split`

    Returns:
        List
    """

    msg = 'Building train-test split'
    logger.info(msg)

    X_train, X_test, y_train, y_test = train_test_split(
            Xy.drop(target, 1), Xy[target], **split_kwargs
            )

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    try:
        make_train_test_split(baking_cookies.project_dir)
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e
