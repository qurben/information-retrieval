import pandas as pd
import numpy as np
import re

df = pd.DataFrame(pd.read_csv('total_data.csv',index_col=0,low_memory=False))
dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

NUMBER_OF_NGRAMS = 4

def suffix_ngrams(string):
    words = re.sub(' +|-|\.', ' ', string).split()

    num_ngrams = min(len(words), NUMBER_OF_NGRAMS)

    ngrams = [' '.join(words[-(i):]) for i in range(1, num_ngrams + 1)]

    return ngrams + [None]*(NUMBER_OF_NGRAMS - len(ngrams))

df = df.fillna('')
df["ngram1"], df["ngram2"], df["ngram3"], df["ngram4"] = zip(*df["Query"].map(suffix_ngrams))

df.to_csv('total_grams_dataset.csv')
