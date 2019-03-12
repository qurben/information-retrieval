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

TRAINING_FILE = 'generated_candidate_head.csv'
FEATURES_FILE = 'training_features.csv'
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


suffix_df = pd.read_csv(SUFFIX_FILE, index_col=0, dtype=object, low_memory=False)
suffix_df = suffix_df.set_index('ngram')

query_df = pd.read_csv(QUERY_FILE, index_col='Query', dtype=object, low_memory=False)


# In[ ]:


dtypes = {
    'Prefix': 'str',
    'Query': 'str',
    'Suffix': 'str',
    'ItemRank': 'str',
    'ClickURL': 'str',
}

# only load index and Query
chunks = pd.read_csv(TRAINING_FILE, dtype=dtypes, low_memory=False, chunksize=CHUNK_SIZE)

# Count the number of chunks in this file
num_chunks = int(sum(1 for row in open(TRAINING_FILE, 'r')) / CHUNK_SIZE) + 1
chunk_id = iter(range(1, num_chunks+1))


# In[ ]:


with open(FEATURES_FILE, 'w') as the_file: the_file.write(',Prefix,Query,Suffix,ngramfreq_1,ngramfreq_2,ngramfreq_3,ngramfreq_4,ngramfreq_5,ngramfreq_5,query_freq,prefix_len_c,prefix_len_w,suffix_len_c,suffix_len_w,len_c,len_w,end_space\n')

for df in chunks:
    print("Processing chunk {} of {}".format(next(chunk_id), num_chunks), end="\r")
    
    # Any empty query is not interesting
    df.dropna(inplace=True)

    df = df.reset_index(drop=True)
    
    df[ngram_cols] = df.Query.apply(ngram_apply, suffix_df=suffix_df)
    df['query_freq'] = df.Query.apply(query_freq, query_df=query_df)
    df['prefix_len_c'] = df.Prefix.str.len()
    df['prefix_len_w'] = df.Prefix.str.split().str.len()
    df['suffix_len_c'] = df.Suffix.str.len()
    df['suffix_len_w'] = df.Suffix.str.split().str.len()
    df['len_c'] = df.Query.str.len()
    df['len_w'] = df.Query.str.split().str.len()
    df['end_space'] = df.Prefix.str.endswith(' ').astype(int)
    
    
    df.to_csv(FEATURES_FILE, mode='a', header=False)

