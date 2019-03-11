import pandas as pd
import dask.dataframe as dd
import re
import glob
import os
import os.path
import uuid

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

    intermediary_files = os.path.join('.tmp', '{}_*.csv'.format(uuid.uuid1()))
    
    normalize_df(df).to_csv(intermediary_files, index=False)
    concat_csv(intermediary_files, out_file)

def concat_csv(infiles, out, rm=True):
    filenames = glob.glob(infiles)
    with open(out, 'w') as outfile:
        with open(filenames[0]) as infile:
            outfile.write(next(infile))

        for fname in filenames:
            with open(fname) as infile:
                next(infile)
                for line in infile:
                    outfile.write(line)
            
            if rm: os.remove(fname)


if __name__ == '__main__':
    normalize_csv(IN_FILE, OUT_FILE)