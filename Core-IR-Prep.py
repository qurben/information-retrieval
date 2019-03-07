#!/usr/bin/env python
# coding: utf-8

# # Prep the data
# 
# Extract all the files from `AOL_search_data_leak_2006.zip` and concat them into a single file.

# In[ ]:


import zipfile
import shutil
import gzip
import glob
import os.path
import pandas

DATA_ZIP_FILE = 'AOL_search_data_leak_2006.zip'
DATA_DIR = 'AOL-user-ct-collection'


# In[ ]:


## Check if the zip file is in this directory

if not os.path.isfile(DATA_ZIP_FILE):
    raise Exception(DATA_ZIP_FILE + ' not found.')


# In[ ]:


## Extract zip file

archive = zipfile.ZipFile(DATA_ZIP_FILE)
archive.extractall()
archive.close()

shutil.rmtree('__MACOSX', ignore_errors=True)


# In[ ]:


## Extract gz files inside zip file

gz_files = glob.glob(DATA_DIR + '/*.gz')
for gz_filename in gz_files:
    txt_filename = gz_filename[:-3]
    with gzip.open(gz_filename, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[ ]:


## Concat all files into a single file

txt_files = glob.glob(DATA_DIR + '/user*.txt')

dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

files = (pandas.read_csv(f, sep="\t", dtype=dtypes) for f in txt_files)
    
frame = pandas.concat(files, ignore_index=True)

frame.sort_values('QueryTime', inplace=True)

frame.to_csv('total_data.csv')


# In[ ]:




