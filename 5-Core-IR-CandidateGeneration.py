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
BASE_FILE = 'popular_query.csv'
OUT_FILE = 'generated_candidate.csv'

CHUNK_SIZE = 10000
N_POPULAR_SUFFIX = 1000


# In[ ]:


suffix_df = pd.read_csv(POPULAR_SUFFIX_FILE, index_col=0, nrows=N_POPULAR_SUFFIX)


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
    return list(suffix_df[suffix_df.ngram.str.startswith(end_term)].ngram)
    
def apply_end_term(row):
    candidates = []        
    query = original_query = row.iloc[0].Query
    
    while query.find(' ') != -1: # There is more than one word
        term = end_term(query)
        suffixes = match_end_term(term)
                
        for suffix in suffixes:
            new_query = query + suffix[len(term):]
            
            if new_query != original_query or True:
                candidates.append({
                    'Prefix': query,
                    'Suffix': suffix,
                    'Query': new_query
                })
        
        query = query[:-1]
    
    return pd.DataFrame(candidates)


# In[ ]:


with open(OUT_FILE, 'w') as the_file: the_file.write('Prefix,Query,Suffix\n')


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
chunks = pd.read_csv(BASE_FILE, index_col=0, dtype=dtypes, usecols=[0, 1], low_memory=False, chunksize=CHUNK_SIZE)

# Count the number of chunks in this file
num_chunks = int(sum(1 for row in open(BASE_FILE, 'r')) / CHUNK_SIZE) + 1
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
    df2 = df.groupby(df.columns.tolist(), group_keys=False).apply(apply_end_term)
    
    df2.to_csv(OUT_FILE, mode='a', header=False, index=False)

