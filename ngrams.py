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

def ngrams(string, i):
    string = string.replace('-',' ')
    string = string.replace('.',' ')
    string = re.sub(' +', ' ', string)
    suffixes = []
    prefixes = []
    if string == ' ':
        suffixes = [None]*i
    else:
        words = string.split(' ')
        for n in range(1,i+1):
            if len(words) == n:
                suffix = str(string)
                prefix = None
            elif len(words) > n:
                suffix = words[-n:]
                prefix = str(words[0:(len(words)-n)])
            else:
                suffix = None
                prefix = None
            if isinstance(suffix, list):
                suffix = ''.join(suffix)
            suffixes.append(suffix)
            prefixes.append(prefix)
    return prefixes, suffixes


# Test file, i only used the first 3000 lines or something like that
#fn = 'AOL-user-ct-collection/user-ct-test-collection-01.txt'
#df = pd.DataFrame(pd.read_csv(fn, sep="\t", dtype=dtypes))
j = 0
n = 6
suffixes = []
# Loop through each query
for i in range(df['Query'].size):
    # Generate n-grams based on the query and n
    prefix, suffix = ngrams(str(df['Query'].iloc[i]),n)
    if j == 0:
        suffix_arr = np.asarray(suffix)
        j = 1
    else:
        suffix_arr = np.vstack((suffix_arr,np.asarray(suffix)))

df_grams = pd.DataFrame(suffix_arr,columns = range(1,(n+1)))
df_grams.to_csv('./total_grams_dataset.csv')
