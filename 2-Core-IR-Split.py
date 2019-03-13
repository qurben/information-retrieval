#!/usr/bin/env python
# coding: utf-8

# # Splitting data
# 
# Split the data into different parts according to the definition in the paper
# 
# > The query impressions on both the testbeds are divided into four temporally separate partitions
# > (background, training, validation and test). On theAOL testbed we use all the data from 1 March,
# > 2006 to 30 April, 2006 as the background data. We sample queries from the next two weeks for
# > training, and from each of the following two weeks for validation and test, respectively. On the
# > Bing testbed we sample data from the logs from April, 2015 and use the first week of data for
# > background, the second week for training, the third for validation and the fourth for testing. We
# > normalize all the queries ineach of these datasets by removing any punctuation characters and
# > converting them to lower case.

# In[ ]:


import pandas as pd
import os.path
from util.normalize import normalize_csv

CHUNK_SIZE = 100000
IN_FILE = 'total_data.csv'


# In[ ]:


if os.path.isfile('background.csv'):
    name = input('Output file already exists, do you want to continue? (y/n): ')
    if name != 'y': exit()


# In[ ]:


data_start = "2006-03-01 00:00:00"
background_end = "2006-03-03 23:59:59"
training_end = "2006-03-05 23:59:59"
validation_end = "2006-03-06 23:59:59"
test_end = "2006-03-08 23:59:59"

dtypes = {
    'Index': 'int64',
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

files = ['background.csv', 'training.csv', 'validation.csv', 'test.csv']


# In[ ]:


with open(IN_FILE, 'r') as in_file:
    header = in_file.readline()

for file in files:
    with open(file, 'w') as the_file:
        the_file.write(header)


# In[ ]:


num_chunks = int(sum(1 for row in open(IN_FILE, 'r')) / CHUNK_SIZE) + 1
chunks = pd.read_csv(IN_FILE, dtype=dtypes, index_col='Index', chunksize=CHUNK_SIZE)
chunk_id = iter(range(1, num_chunks+1))


# In[ ]:


for df in chunks:
    print("Processing chunk {} of {}".format(next(chunk_id), num_chunks), end="\r")

    background = df[(df['QueryTime'] > data_start) & (df['QueryTime'] < background_end)]
    training = df[(df['QueryTime'] > background_end) & (df['QueryTime'] < training_end)]
    validation = df[(df['QueryTime'] > training_end) & (df['QueryTime'] < validation_end)]
    test = df[(df['QueryTime'] > validation_end) & (df['QueryTime'] < test_end)]

    background.to_csv('background.csv', mode='a', header=False)
    training.to_csv('training.csv', mode='a', header=False)
    validation.to_csv('validation.csv', mode='a', header=False)
    test.to_csv('test.csv', mode='a', header=False)


# In[ ]:


normalize_csv('background.csv', 'background_normalized.csv')
normalize_csv('training.csv', 'training_normalized.csv')
normalize_csv('validation.csv', 'validation_normalized.csv')
normalize_csv('test.csv', 'test_normalized.csv')

