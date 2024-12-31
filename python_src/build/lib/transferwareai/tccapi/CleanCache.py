import json
import pandas as pd
import polars as pl


""" 
We identified an issue with the cache which means that it could
not be read into a polars df without throwing an error. The thrown error is
as follows: polars.exceptions.ComputeError: extra key in struct data: maker
This meant that there was a key trying to be read into the df that is different from the
already inferred schema. 
 
The source of this error comes from some makers were being read in as
full patterns and not just components of a pattern. Since maker has a different schema than
a full pattern, it threw the error. 

We will identify and remove these maker keys by reading them into a pandas df and removing
them before rewriting the cache file

** Since this error originates from how TCC structures the pattern information, this script
may need to be updated to solve additional errors
"""

def clean_cache(filename):
    df = pd.read_json(filename)

    if 'maker' in df.keys():
        problem_indexes = df[df['maker'].notna()].index

        new_cache = []
        with open (filename, "r") as f:
            temp_cache = json.load(f)

        i = 0
        for record in temp_cache:
            if i not in problem_indexes:
                new_cache.append(record)
                i += 1

        with open(filename, "w") as f:
           json.dump(new_cache, f, indent=2)

        # For debugging purposes -- uncomment to check that things are read in correctly
        # cache = pl.read_json("cache.json")
        # print(cache.head())

    return

