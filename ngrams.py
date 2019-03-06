import pandas as pd
from nltk import ngrams

#df = pd.DataFrame(pd.read_csv('total_data.csv',index_col=0,low_memory=False))
dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}

# Test file, i only used the first 3000 lines or something like that
fn = 'AOL-user-ct-collection/user-ct-test-collection-01.txt'
df = pd.DataFrame(pd.read_csv(fn, sep="\t", dtype=dtypes))
j = 0
grams = []
# The paper uses n=1..6 for the n-grams
for n in range(1,7):
    # Loop through each query
    for i in range(df['Query'].size):
        # Generate n-grams based on the query and n
        gram_gens = ngrams(str(df['Query'].iloc[i]).split(),n)
        # Get the n-grams and append them to a list
        for gram_gen in gram_gens:
            grams.append(' '.join(gram_gen))
    # Create pandas dataframe
    if j == 0:
        df_grams = pd.DataFrame(grams, index = range(len(grams)), columns=[str(n)])
        j = 1
    else:
        temp = pd.DataFrame(grams, index = range(len(grams)),columns=[str(n)])
        df_grams = pd.concat([df_grams, temp], axis=1)
    grams = []

# Print the most frequent 1-grams
print(df_grams['1'].value_counts())
