#!/usr/bin/env python
# coding: utf-8

# # Generate ngrams for every row

# In[ ]:


import pandas as pd
import numpy as np
import re
import os
import os.path

IN_FILE = 'background.csv'
OUT_FILE = 'total_data_ngrams.csv'
CHUNK_SIZE = 100000
NUMBER_OF_NGRAMS = 4


# In[ ]:


def apply_suffix_ngrams(query):
#     query = row.iloc[0]["Query"]
    if pd.isna(query): query = ''
    return pd.Series(suffix_ngrams(query))

def suffix_ngrams(string):
    words = re.sub(r' +|-|\.', ' ', ' ' + string).split()
    num_ngrams = min(len(words), NUMBER_OF_NGRAMS)
    
    if num_ngrams == 0: return []

    for i in range(num_ngrams): yield ' '.join(words[(-1-i):])


# In[ ]:


ngram_cols = ['ngram{}'.format(n+1) for n in range(NUMBER_OF_NGRAMS)]

with open(OUT_FILE, 'w') as the_file, open(IN_FILE, 'r') as in_file:
    line = in_file.readline()[:-1]
    the_file.write('id,ngram\n')


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
    df.reset_index(inplace=True)
    
    df = df.Query.apply(apply_suffix_ngrams)         .merge(df, right_index = True, left_index = True)         .drop(['Query'], axis=1)         .melt(id_vars = ['index'], value_name = "ngram")         .drop(["variable", "index"], axis = 1)         .dropna()
    
    df.to_csv(OUT_FILE, mode='a', header=False)

