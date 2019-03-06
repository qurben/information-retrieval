import pandas as pd
import os

txt_files = os.listdir('./AOL-user-ct-collection/')

for txt_file in txt_files:
    if txt_file.endswith('.txt'):
        pass
    else:
        txt_files.remove(txt_file)

dtypes = {
    'AnonID': 'str',
    'Query': 'str',
    'QueryTime': 'str',
    'ItemRank': 'str',
    'ClickUrl': 'str',
}
print(txt_files)
files = (pd.read_csv(os.path.join('AOL-user-ct-collection/'+str(f)), sep="\t", dtype=dtypes) for f in txt_files)

frame = pd.concat(files, ignore_index=True)

frame.sort_values('QueryTime', inplace=True)

print(frame)
frame.to_csv('total_data.csv')
