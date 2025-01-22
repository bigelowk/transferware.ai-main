import json
import pandas as pd
import polars as pl


""" 
1/20/25 - We identified some issues with the cache which means that it could
not be read into a polars df without throwing an error. 

The two errors we encountered are as follows: 
1) print_process data type error: polars.exceptions.ComputeError: error deserializing value "Static(Bool(false))" as struct
We compared the expected schema to the cache and found that in some cases, print_process had the value 'false' instead
of the expected dictionary structure. 

2) maker feature: polars.exceptions.ComputeError: extra key in struct data: maker
This meant that there was a key trying to be read into the df that is different from the
already inferred schema. The source of this error comes from some makers were being read in as
full patterns and not just components of a pattern. Since maker has a different schema than
a full pattern, it threw the error. We identified and removed these maker keys by reading them into a pandas df and removing
them before rewriting the cache file

** Since this error originates from how TCC structures the pattern information, this script
may need to be updated to solve additional errors
"""

def clean_cache(filename):
    df = pd.read_json(filename)

    # remove records for maker -- these are not records for specific patterns but rater just a singular makers mark
    problem_indexes = df[df['maker'].notna()].index if 'maker' in df.keys() else []

    new_cache = []
    with open (filename, "r") as f:
        temp_cache = json.load(f)

    for i in range(len(temp_cache)):
        if i not in problem_indexes:
            # replace anywhere that there is False with an empty dictionary to make sure that the data type of the column is uniform
            if isinstance(temp_cache[i]['print_process'], bool):    # this will either be false or a dictionary
                temp_cache[i]['print_process'] = {}
            new_cache.append(temp_cache[i])

    with open(filename, "w") as f:
       json.dump(new_cache, f, indent=2)

    return


if __name__ == "__main__":
    clean_cache("bad_cache.json")

    cache = pl.read_json("cache.json", infer_schema_length=100)

    # use this to check if row that previously said "false" is now {}
    test = cache.filter(pl.col("id") == 81193)
    print(test)
    pass

