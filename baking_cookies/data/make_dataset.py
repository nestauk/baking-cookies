# -*- coding: utf-8 -*-
import click
import os
import logging
from dotenv import find_dotenv, load_dotenv
import baking_cookies
import pandas as pd
from baking_cookies.data.gtr import make_gtr
from baking_cookies.data.sql import get_engine


logger = logging.getLogger(__name__)

@click.command()
@click.option('--first-time/--not-first-time', default=False)
def main(first_time):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    Usage: `python make_dataset.py --first-time`

    Args:
        first_time (bool, optional):
            If True, fetches files from mysql and saves to `data/raw`
            Defaults to False.
    """
    config = baking_cookies.config

    if first_time:
        fout_gtr_projects = f"{project_dir}/data/raw/gtr_projects.csv"
        logger.info('Fetching raw data from MySQL')
        engine = get_engine(os.getenv('sql_config_path'))
        projects = pd.read_sql_table('gtr_projects', engine)
        projects.to_csv(fout_gtr_projects, index=False)
        logger.info(f'Fetched and saved raw data from MySQL to {fout_gtr_projects}')

    logger.info('Making gtr dataset')
    make_gtr(project_dir / 'data',
             config['data']['gtr']['usecols'],
             config['data']['gtr']['nrows'],
             config['data']['gtr']['min_chars'],
             config['data']['gtr']['min_length'])


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = baking_cookies.project_dir

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        logger.exception(e, stack_info=True)
        raise e
