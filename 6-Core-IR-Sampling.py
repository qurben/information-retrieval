#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import dask.dataframe as dd

from util.dask import to_csv

CHUNK_SIZE = 1000

dtypes = {
    'Index': 'int64',
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickURL': 'str',
}


# In[ ]:


def sample_query(query):
    yield query
    
    for i in range(1, len(query)):
        yield query[:-i]
        
def extract_suffix(row):
    query = row.Query
    prefix = row.prefix


# In[ ]:


def sample_dataset(in_file, out_file):
    chunks = pd.read_csv(in_file, index_col='Index', usecols=['Index','Query'], dtype=dtypes, chunksize=CHUNK_SIZE)
    
    first = True
    # Count the number of chunks in this file
    num_chunks = int(sum(1 for row in open(in_file, 'r')) / CHUNK_SIZE) + 1
    chunk_id = iter(range(1, num_chunks+1))
    
    with open(out_file, 'w') as f: f.write('')

    for df in chunks:
        print("Processing chunk {} of {}".format(next(chunk_id), num_chunks), end="\r")

        # Any empty query is not interesting
        df.dropna(inplace=True)

        # 1. Apply suffix_ngrams, creates a list of ngrams for each row
        # 2. Apply pd.Series, creates a series for this list
        # 3. Merge the applied series with the dataframe
        # 4. Reset the index, make it available for selection
        # 5. Melt with the index and Query as id, this flattens the ngrams list
        # 6. Drop the variable column, they are not interesting anymore
        df = df.Query.apply(sample_query).apply(pd.Series)             .merge(df, right_index = True, left_index = True)             .reset_index()             .melt(id_vars = ['Index', 'Query'], value_name = 'Prefix')             .drop(['variable'], axis = 1)
        
        df['Suffix'] = ''

        df.to_csv(out_file, mode='a', header=first, index=False)
        first = False


# In[ ]:


sample_dataset('test_normalized.csv', 'test_sampled.csv')
sample_dataset('validation_normalized.csv', 'validation_sampled.csv')

