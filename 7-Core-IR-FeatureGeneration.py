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
import dask.dataframe as dd
import nltk
import numpy as np
from sklearn.datasets import dump_svmlight_file

from util.dask import to_csv

SUFFIX_FILE = 'popular_suffix.csv'
QUERY_FILE = 'popular_query.csv'

CHUNK_SIZE = 10000


# ## Ngram based features

# In[ ]:


ngram_cols = ['ngramfreq_{}'.format(i) for i in range(1,7)]

def ngram_apply(query, suffix_df):
    words = query.split()
    ngrams = []
        
    for n in range(1,7):
        s = 0
        for ngram in nltk.ngrams(words, n):
            try:
                counts = int(suffix_df.at[' '.join(ngram), 'counts'])
            except KeyError:
                counts = 0

            s += counts
            
        ngrams.append(s)
    
    return pd.Series(ngrams)

def query_freq(query, query_df):
    try:
        return int(query_df.at[query, 'counts'])
    except KeyError:
        return 0


# In[ ]:


suffix_df = pd.read_csv(SUFFIX_FILE, index_col='ngram', dtype=object, low_memory=False)
query_df = pd.read_csv(QUERY_FILE, index_col='Query', dtype=object, low_memory=False)


# In[ ]:


def calculate_features(in_file, svmlight_file):
    chunks = pd.read_csv(in_file, dtype=object, low_memory=False, chunksize=30000)

    df2 = pd.read_csv(in_file, dtype=object, low_memory=False)
    df2 = df2.fillna('')
    df2 = df2.reset_index(drop=True)

    qids = df2.Prefix.unique()
    qids = pd.Series(np.arange(len(qids)), qids)
        
    num_chunks = int(sum(1 for row in open(in_file, 'r')) / CHUNK_SIZE) + 1
    chunk_id = iter(range(1, num_chunks+1))
    
    for df in chunks:
        print("Processing chunk {} of {}".format(next(chunk_id), num_chunks), end="\r")
                
        df = df.fillna('')
        
        ## Generate features for each row
        df['prefix_id'] = df.Prefix.apply(lambda x : qids.loc[x])
        df[ngram_cols] = df.Query.apply(ngram_apply, suffix_df=suffix_df)
        df['query_freq'] = df.Query.apply(query_freq, query_df=query_df)
        df['is_relevant'] = df['query_freq'].apply(lambda x : 1 if x > 0 else 0)
        df['prefix_len_c'] = df.Prefix.str.len()
        df['prefix_len_w'] = df.Prefix.str.split().str.len()
        df['suffix_len_c'] = df.Suffix.str.len()
        df['suffix_len_w'] = df.Suffix.str.split().str.len()
        df['len_c'] = df.Query.str.len()
        df['len_w'] = df.Query.str.split().str.len()
        df['end_space'] = df.Prefix.str.endswith(' ').astype(int)

        df = df.drop('Prefix', axis=1)
        df = df.drop('Query', axis=1)
        df = df.drop('Suffix', axis=1)

        df = df.sort_values('prefix_id')

        data = df.drop('prefix_id', axis=1).drop('is_relevant', axis=1).drop('Index', axis=1)

        dump_svmlight_file(data, y=df['is_relevant'], query_id=df['prefix_id'], f=svmlight_file.format(index))


# In[ ]:


calculate_features('training_sampled.csv', 'training_features.svmlight.txt', 'training_features.csv')
calculate_features('test_sampled.csv', 'test_features_{}.svmlight.txt', 'test_features.csv')
calculate_features('validation_sampled.csv', 'validation_features_{}.svmlight.txt')

