#!/usr/bin/env python
# coding: utf-8

# # Generate ngrams for every row

# In[ ]:


import pandas as pd
import numpy as np
import re
import os
import os.path

IN_FILE = 'total_data.csv'
OUT_FILE = 'total_data_ngrams.csv'
CHUNK_SIZE = 100000
NUMBER_OF_NGRAMS = 2


# In[ ]:


def apply_suffix_ngrams(row):
    query = row["Query"]
    if pd.isna(query): query = ''
    return pd.Series(suffix_ngrams(query))

def suffix_ngrams(string):
    words = re.sub(r' +|-|\.', ' ', string).split()
    num_ngrams = min(len(words), NUMBER_OF_NGRAMS)

    for i in range(num_ngrams): yield ' '.join(words[(-1-i):])
    for _ in range(NUMBER_OF_NGRAMS - num_ngrams): yield None


# In[ ]:


ngram_cols = ['ngram{}'.format(n+1) for n in range(NUMBER_OF_NGRAMS)]

with open(OUT_FILE, 'w') as the_file, open(IN_FILE, 'r') as in_file:
    line = in_file.readline()[:-1]
    the_file.write(line + ',' + ','.join(ngram_cols) + '\n')


# In[ ]:


chunks = pd.read_csv(IN_FILE, index_col=0, low_memory=False, chunksize=CHUNK_SIZE)
dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

# Count the number of chunks in this file
num_chunks = int(sum(1 for row in open(IN_FILE, 'r')) / CHUNK_SIZE) + 1
chunk_ids = iter(range(1, num_chunks+1))

for df in chunks:
    print("Processing chunk {} of {}".format(next(chunk_ids), num_chunks), end="\r")
    df[ngram_cols] = df.apply(apply_suffix_ngrams, axis=1)
    df.to_csv(OUT_FILE, mode='a', header=False)

