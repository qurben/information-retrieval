import pandas as pd
import dask.dataframe as dd
import re
import glob
import os
import os.path
import uuid

from .dask import to_csv

IN_FILE = 'validation.csv'
OUT_FILE = 'validation_normalized.csv'

def normalize_query(query, axis):
    return re.sub('-|\'|\.', '', query).strip()

def normalize_df(df):
    df['Query'] = df.Query.dropna().apply(normalize_query, axis=1, meta=('str'))
    return df

def normalize_csv(in_file, out_file):
    if not os.path.exists('.tmp'):
        os.makedirs('.tmp')

    df = dd.read_csv(in_file, dtype=object)

    df = normalize_df(df)

    to_csv(df, out_file)
