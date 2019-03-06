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

# In[6]:


import pandas
from dateutil import parser


# In[ ]:


data_start = "2006-03-01 00:00:00"
background_end = "2006-04-30 23:59:59"
training_end = "2006-05-14 23:59:59"
validation_end = "2006-05-21 23:59:59"
test_end = "2006-03-28 23:59:59"

dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}


# In[ ]:


df = pandas.read_csv('data.csv', dtype=dtypes)
df = df.set_index(df['QueryTime'])


# In[ ]:


background = df[data_start:background_end]
background.to_csv('background.csv')


# In[ ]:


training = df[background_end:training_end]
training.to_csv('training.csv')


# In[ ]:


validation = df[training_end:validation_end]
validation.to_csv('validation.csv')


# In[ ]:


test = df[validation_end:test_end]
test.to_csv('test.csv')

