import json
import pandas as pd
import polars as pl


""" 
1/20/25 - We identified some issues with the cache which means that it could
not be read into a polars df without throwing an error. 

The two errors we encountered are as follows: 
1) print_process data type error: polars.exceptions.ComputeError: error deserializing value "Static(Bool(false))" as struct
We compared the expected schema to the cache and found that in some cases, print_process had the value 'false' instead
of the expected dictionary structure. Similar errors will occur when the columns have inconsistent data types

2) maker feature: polars.exceptions.ComputeError: extra key in struct data: maker
This meant that there was a key trying to be read into the df that is different from the
already inferred schema. The source of this error comes from some makers were being read in as
full patterns and not just components of a pattern. Since maker has a different schema than
a full pattern, it threw the error. We identified and removed these maker keys by reading them into a pandas df and removing
them before rewriting the cache file

To solve these issues, we will only take the relevant selection of the columns.
"""

def clean_cache(filename):
    df = pd.read_json(filename)

    # only take the columns that are necessary for analysis
    df_only_necessary_columns = df[['id', 'url', 'name', 'pattern_number', 'title', 'alternate_names',
                                    'category', 'images']]


    json_str = df_only_necessary_columns.to_json(orient='records', indent=2)
    # fix how python writes the urls
    json_str = json_str.replace(r'\/', '/')

    with open(filename, 'w') as file:
        file.write(json_str)

    return


if __name__ == "__main__":

    # clean the cache file without having to engage the training script

    # insert the cache file you want to clean. Should be in the same format as is what is created when the cache
    # is created/updated in the training script
    cache_file = "cache.json"
    clean_cache(cache_file)

    cache = pl.read_json(cache_file)

    # use this to look at what the data looks like
    test = cache.filter(pl.col("id") == 81193)
    print(test)

