#!/usr/bin/env python
# coding: utf-8

# # Candidate Generation
# 
# > for a given prefix we extract the end-term as shown in Figure 1. We match all the suffixes that
# > start with the end-term from our precomputed set. These selected suffixes are appended to the
# > prefix to generate synthetic suggestion candidates. For example, the prefix "cheap flights fro"
# > is matched with the suffix "from seattle" to generate the candidate "cheap flights from seattle".
# > Note that many of these synthetic suggestion candidates are likely to not have been observed by
# > the search engine before. We merge these synthetic suggestions with the set of candidates
# > selected from the list of historically popular queries. This combined set of candidates is used
# > for ranking as we will describe in Sec 4.

# In[ ]:


import pandas as pd
import dask.dataframe as dd

POPULAR_SUFFIX_FILE = 'popular_suffix.csv'
BASE_FILE = 'training_normalized.csv'
OUT_FILE = 'training_sampled.csv'

CHUNK_SIZE = 10000
N_POPULAR_SUFFIX = 1000


# In[ ]:


suffix_df = pd.read_csv(POPULAR_SUFFIX_FILE, index_col='ngram', nrows=N_POPULAR_SUFFIX)


# In[ ]:


def end_term(query):
    """
    end_term('cheapest flight fro') = 'fro'
    end_term('cheapest flight from') = 'from'
    end_term('cheapest flight from ') = 'from '
    end_term('cheapest flight from n') = 'n'
    """
    if query.endswith(' '):
        return query[query[:-1].rfind(' ')+1:]
    else:
        return query[query.rfind(' ')+1:]

def match_end_term(end_term):
    return list(suffix_df[suffix_df.index.str.startswith(end_term)].index)
    
def apply_end_term(row):
    query = original_query = row.Query.iat[0]
        
    candidates = [{
        'Prefix': query,
        'Suffix': '',
        'Query': query
    }]
    
    while query.find(' ') != -1: # There is more than one word
        term = end_term(query)
        suffixes = match_end_term(term)
                
        for suffix in suffixes[:10]:
            new_query = query + suffix[len(term):]
            
            if new_query != original_query or True:
                candidates.append({
                    'Prefix': query,
                    'Suffix': suffix,
                    'Query': new_query
                })
        
        query = query[:query.rfind(' ')]
    
    return pd.DataFrame(candidates)


# In[ ]:


with open(OUT_FILE, 'w') as the_file: the_file.write('Index,Prefix,Query,Suffix\n')


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
df = pd.read_csv(BASE_FILE, index_col='Index', dtype=dtypes, low_memory=False)


# In[ ]:


# Any empty query is not interesting
df.dropna(inplace=True)

df = df.groupby(df.columns.tolist(), group_keys=False).apply(apply_end_term)
df = df.reset_index()
df = df.drop('index', axis=1) # drop de oude index

df.to_csv(OUT_FILE, mode='a', header=False)

