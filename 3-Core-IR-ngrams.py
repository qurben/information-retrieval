#!/usr/bin/env python
# coding: utf-8

# # Generate ngrams for every row
# 
# Generate the ngrams from the background data

# In[ ]:


import pandas as pd
import dask.dataframe as dd
import re

IN_FILE = 'background.csv'
SUFFIX_FILE = 'total_data_ngrams.csv'
POPULAR_SUFFIX_FILE = 'popular_suffix.csv'
CHUNK_SIZE = 10000
MAX_NUMBER_OF_NGRAMS = 3


# In[ ]:


def suffix_ngrams(string):
    string = re.sub('-|\'', '', string) # Remove - and '
    string = re.sub('  ', ' ', string) # Remove double space (can be caused by removing a -)
    words = (' ' + string).split()
    num_ngrams = min(len(words), MAX_NUMBER_OF_NGRAMS)
    
    for i in range(num_ngrams): yield ' '.join(words[(-1-i):])


# In[ ]:


with open(SUFFIX_FILE, 'w') as the_file: the_file.write('')


# In[ ]:


dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickURL': 'str',
}

# only load index and Query
chunks = pd.read_csv(IN_FILE, index_col=0, dtype=dtypes, usecols=[0, 2], low_memory=False, chunksize=CHUNK_SIZE)

# Count the number of chunks in this file
num_chunks = int(sum(1 for row in open(IN_FILE, 'r')) / CHUNK_SIZE) + 1
chunk_id = iter(range(1, num_chunks+1))


# In[ ]:


for df in chunks:
    print("Processing chunk {} of {}".format(next(chunk_id), num_chunks), end="\r")
    
    # Any empty query is not interesting
    df.dropna(inplace=True)
    
    # 1. Apply suffix_ngrams, creates a list of ngrams for each row
    # 2. Apply pd.Series, creates a series for this list
    # 3. Merge the applied series with the dataframe
    # 3. Drop the Query column, we don't need it anymore
    # 4. Reset the index, make it available for selection
    # 5. Melt with the index as id, this flattens the ngrams list
    # 6. Drop the variable and index columns, they are not interesting anymore
    # 7. Drop any empty values (any rows with less than NUMBER_OF_NGRAMS ngrams.
    df = df.Query.apply(suffix_ngrams).apply(pd.Series)         .merge(df, right_index = True, left_index = True)         .drop(['Query'], axis=1)         .reset_index()         .melt(id_vars = ['index'], value_name = "ngram")         .drop(['variable', 'index'], axis = 1)         .dropna()
    
    df.to_csv(SUFFIX_FILE, mode='a', header=False, index=False)


# In[ ]:


df = pd.read_csv(SUFFIX_FILE, header=None, names=['ngram'])
df = dd.from_pandas(df, chunksize=CHUNK_SIZE)
df = df.groupby('ngram').agg('size').compute()        .reset_index(name='counts')        .sort_values('counts', ascending=False)
df.to_csv(POPULAR_SUFFIX_FILE)

