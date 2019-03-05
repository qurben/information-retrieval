#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import shutil
import gzip
import glob
import os.path
import pandas

DATA_ZIP_FILE = 'AOL_search_data_leak_2006.zip'
DATA_DIR = 'AOL-user-ct-collection'


# In[2]:


## Check if the zip file is in this directory

if not os.path.isfile(DATA_ZIP_FILE):
    raise Exception(DATA_ZIP_FILE + ' not found.')


# In[3]:


## Extract zip file

archive = zipfile.ZipFile(DATA_ZIP_FILE)
archive.extractall()
archive.close()

shutil.rmtree('__MACOSX', ignore_errors=True)


# In[4]:


## Extract gz files inside zip file

gz_files = glob.glob(DATA_DIR + '/*.gz')
for gz_filename in gz_files:
    txt_filename = gz_filename[:-3]
    with gzip.open(gz_filename, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[11]:


## Concat all files into a single file

txt_files = glob.glob(DATA_DIR + '/user*.txt')

dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

files = (pandas.read_csv(f, sep="\t", dtype=dtypes, parse_dates=[2]) for f in txt_files)
    
frame = pandas.concat(files, axis=0, ignore_index=True)

frame.sort_values(by=['QueryTime'])

frame.to_csv('data.csv')


# In[ ]:




