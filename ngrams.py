import pandas as pd
import numpy as np
import re
import os
import os.path

IN_FILE = 'total_data.csv'
OUT_FILE = 'total_grams_dataset.csv'
CHUNK_SIZE = 100000

if os.path.exists(OUT_FILE): os.remove(OUT_FILE)

with open(OUT_FILE, 'a') as the_file, open(IN_FILE, 'r') as in_file:
    line = in_file.readline()[:-1]
    the_file.write(line + ',ngram1,ngram2,ngram3,ngram4,ngram5,ngram6\n')

num_chunks = int(sum(1 for row in open(IN_FILE, 'r')) / CHUNK_SIZE) + 1

chunks = pd.read_csv(IN_FILE,index_col=0,low_memory=False, chunksize=CHUNK_SIZE)
dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

NUMBER_OF_NGRAMS = 6

def suffix_ngrams(string):
    words = re.sub(' +|-|\.', ' ', string).split()

    num_ngrams = min(len(words), NUMBER_OF_NGRAMS)

    ngrams = [' '.join(words[-(i):]) for i in range(1, num_ngrams + 1)]

    return ngrams + [None]*(NUMBER_OF_NGRAMS - len(ngrams))

chunk_id = 1


for df in chunks:
    print("Processing chunk " + str(chunk_id) + " of " + str(num_chunks))
    chunk_id = chunk_id+1
    df = df.fillna('')
    df["ngram1"], df["ngram2"], df["ngram3"], df["ngram4"], df["ngram5"], df["ngram6"] = zip(*df["Query"].map(suffix_ngrams))
    df.to_csv(OUT_FILE, mode='a', header=False)