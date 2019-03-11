#!/usr/bin/env python
# coding: utf-8

# # Popular Query mining
# 
# > For candidate generation, both the list of popular queries and suffixes are mined from the
# > background portion of the two testbeds. We use 724,340 and 1,040,674 distinct queries on the AOL
# > testbed and the Bing testbed, respectively, as the set of full-query candidates
# 
# We need to find the 724340 most popular distinct queries from the background data.

# In[ ]:


import dask.dataframe as dd
import pandas as pd

NUM_QUERIES = 724340
CHUNK_SIZE = 10000
IN_FILE = 'background_normalized.csv'
OUT_FILE = 'popular_query.csv'


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
df = pd.read_csv(IN_FILE, dtype=dtypes, usecols=[0, 2])
df = dd.from_pandas(df, chunksize=CHUNK_SIZE)

print('CSV loaded')

df = df.groupby('Query').agg('size').compute().reset_index(name='counts')

print('Group by finished')

df = df.sort_values('counts', ascending=False)
df = df.head(NUM_QUERIES)

df.to_csv(OUT_FILE)


# In[ ]:




