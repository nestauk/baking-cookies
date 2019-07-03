import logging
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from configparser import ConfigParser
import baking_cookies

def get_engine(config_path, database="production", **engine_kwargs):
    '''Get a SQL alchemy engine from config'''
    cp = ConfigParser()
    cp.read(config_path)
    cp = cp["client"]
    url = URL(drivername="mysql+pymysql", database=database,
              username=cp["user"], host=cp["host"], password=cp["password"])
    engine = create_engine(url, **engine_kwargs)
    engine.execution_options(stream_results=True)
    return engine
