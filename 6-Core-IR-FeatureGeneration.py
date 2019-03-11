#!/usr/bin/env python
# coding: utf-8

# # Feature Generation
# 
# ## N-gram based features
# 
# > From the set of all n-grams G in a candidate suggestion we compute the
# > n-gram frequency features n gram _freq<sub>i</sub>_ (for _i_ = 1 to 6).
# 
# ## Other features
# 
# - frequency of the candidate query in the historical logs
# - length of the prefix in characters
# - length of the prefix in words
# - length of the suffix in characters
# - length of the suffix in words
# - length of the full suggestion
# - length of the full words
# - whether a prefix ends with a space (boolean)

# In[ ]:


import pandas as pd
import nltk

TRAINING_FILE = 'short.csv'
FEATURES_FILE = 'features.csv'

CHUNK_SIZE = 10000


# ## Ngram based features

# In[ ]:


ngram_cols = ['ngramfreq_{}'.format(i) for i in range(1,7)]

def ngram_apply(row):
    query = row.Query
    words = query.split()
    ngrams = []
    
    print([a for a in nltk.ngrams(words, 2)])
    
    for n in range(1,7):
        ngrams.append(sum(1 for _ in nltk.ngrams(words, n)))
    
    return pd.Series(ngrams)


# In[ ]:


dtypes = {
    'Index': 'int64',
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickURL': 'str',
}

# only load index and Query
chunks = pd.read_csv(TRAINING_FILE, index_col=0, dtype=dtypes, usecols=[0, 2], low_memory=False, chunksize=CHUNK_SIZE)

# Count the number of chunks in this file
num_chunks = int(sum(1 for row in open(TRAINING_FILE, 'r')) / CHUNK_SIZE) + 1
chunk_id = iter(range(1, num_chunks+1))


# In[ ]:


for df in chunks:
    print("Processing chunk {} of {}".format(next(chunk_id), num_chunks), end="\r")
    
    # Any empty query is not interesting
    df.dropna(inplace=True)

    df = df.reset_index(drop=True)
    
    df[ngram_cols] = df.apply(ngram_apply, axis=1)
    
    print(df)

