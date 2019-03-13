import os
import os.path
import dask.dataframe as dd
import uuid
import glob

def to_csv(dask_df, out_file):
    if not os.path.exists('.tmp'):
        os.makedirs('.tmp')

    intermediary_files = os.path.join('.tmp', '{}_*.csv'.format(uuid.uuid1()))
    
    dask_df.to_csv(intermediary_files, index=False)
    concat_csv(intermediary_files, out_file)

def concat_csv(infiles, out, rm=True):
    filenames = glob.glob(infiles)
    with open(out, 'w') as outfile:
        with open(filenames[0]) as infile:
            outfile.write(next(infile))

        for fname in filenames:
            with open(fname) as infile:
                next(infile)
                for line in infile:
                    outfile.write(line)
            
            if rm: os.remove(fname)
