import pandas as pd
import numpy as np 

def get_sample_data(dataframe, sample_size): 
    return dataframe.sample(n=sample_size, random_state=1).reset_index(drop=True)

def get_splits(dataframe, reviews_data, target_domain):
    # get the top n% of the data
    dataframe = dataframe.sort_values(by=f"similarity_to_{target_domain}", ascending=False)
    n = int(len(dataframe)*0.3)
    top_n = dataframe.head(n)
    bottom_n = dataframe.tail(n)

    return top_n, bottom_n




