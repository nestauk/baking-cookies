import logging
import ast
import yaml
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots
from pandas import read_csv
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
import baking_cookies
from baking_cookies.models.train_model import _identity

logger = logging.getLogger(__name__)


def make_evaluate(project_dir):
    """Evaluates model metrics on test set.

    Outputs metrics to `models/metrics.txt`

    Args:
        project_dir (str): Project directory path
    """

    config = baking_cookies.config['evaluate']
    target = baking_cookies.config['target']
    logger.info(f'Loaded evaluate config parameters: {config}')

    test_fin = f"{project_dir}/data/processed/gtr_tokenised.csv"
    test_id_fin = f"{project_dir}/data/processed/gtr_leadFunder_id_test.csv"
    clf_fin = f"{project_dir}/models/gtr_{target}.pkl"
    metrics_fout = f"{project_dir}/models/metrics.json"
    confusion_fout = f"{project_dir}/reports/figures/gtr_{target}_confusion_matrix.png"

    x_test, y_test = (
                      # Read data
                      read_csv(test_fin, index_col=0,
                      converters={'processed_documents': ast.literal_eval})
                      # Get train indexes
                      .merge(read_csv(test_id_fin), on='id')
                      # Feature-target split
                      .pipe(lambda x: (x['processed_documents'], x[target]))
                      )

    clf = load(clf_fin)
    logger.info(f"Loaded model")

    # Classification report
    report = classification_report(y_test, clf.predict(x_test),
            output_dict=True)
    logger.info(f"Test classification report: {report}")
    with open(metrics_fout, 'w') as f:
        json.dump({'gtr_clf': report}, f)
    logger.info(f"Saved classifier report to {metrics_fout}")

    # Confusion matrix
    fig, ax = subplots()
    cm = ConfusionMatrix(clf, ax=ax)
    cm.score(x_test, y_test)
    cm.poof(outpath=confusion_fout)


if __name__ == '__main__':
    try:
        make_evaluate(baking_cookies.project_dir)
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e
