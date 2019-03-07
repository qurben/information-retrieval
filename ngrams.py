import pandas as pd
import numpy as np
import re
import os
import os.path

IN_FILE = 'total_data.csv'
OUT_FILE = 'total_data_ngrams.csv'
CHUNK_SIZE = 100000
NUMBER_OF_NGRAMS = 2

ngram_cols = ['ngram{}'.format(n+1) for n in range(NUMBER_OF_NGRAMS)]

if os.path.exists(OUT_FILE): os.remove(OUT_FILE)

with open(OUT_FILE, 'a') as the_file, open(IN_FILE, 'r') as in_file:
    line = in_file.readline()[:-1]
    the_file.write(line + ',' + ','.join(ngram_cols) + '\n')

# Count the number of chunks in this file
num_chunks = int(sum(1 for row in open(IN_FILE, 'r')) / CHUNK_SIZE) + 1

chunks = pd.read_csv(IN_FILE,index_col=0,low_memory=False, chunksize=CHUNK_SIZE)
dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

def sfng(row):
    query = row["Query"]
    if pd.isna(query): query = ''
    return pd.Series(suffix_ngrams(query))

def suffix_ngrams(string):
    words = re.sub(r' +|-|\.', ' ', string).split()
    num_ngrams = min(len(words), NUMBER_OF_NGRAMS)

    for i in range(num_ngrams): yield ' '.join(words[(-1-i):])
    for _ in range(NUMBER_OF_NGRAMS - num_ngrams): yield None

chunk_ids = iter(range(1, num_chunks+1))

for df in chunks:
    print("Processing chunk {} of {}".format(next(chunk_ids), num_chunks), end="\r")
    df[ngram_cols] = df.apply(sfng, axis=1)
    df.to_csv(OUT_FILE, mode='a', header=False)