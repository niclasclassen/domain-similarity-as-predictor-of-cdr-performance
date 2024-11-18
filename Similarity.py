from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
from datasets import load_dataset, load_from_disk
import Data_processing_ as dp

def get_datasets(dataset_name1, dataset_name2):
    dataset1 = load_from_disk(f"Data\{dataset_name1}")
    dataset2 = load_from_disk(f"Data\{dataset_name2}")
    data1 = dp.merge_columns(dataset1)
    data2 = dp.merge_columns(dataset2)

    #generate the embeddings
    emb_data = dp.generate_embeddings(data1)
    emb_data2 = dp.generate_embeddings(data2)

    df1 = emb_data.to_pandas()[["combined", "embeddings"]]
    df2 = emb_data2.to_pandas()[["combined", "embeddings"]]

    return df1, df2

def get_dataset(dataset_name1):
    dataset1 = load_from_disk(f"Data\{dataset_name1}")

    df1 = dataset1.to_pandas()

    return df1

def avg_cosine_similarity(df1_name, df2_name):
    # Calculates the similarity between two datasets
    df1, df2 = get_datasets(df1_name, df2_name)
    embeddings1 = np.array(df1["embeddings"].tolist())
    embeddings2 = np.array(df2["embeddings"].tolist())
    
    # Calculate cosine similarity between every embedding in df1 and every embedding in df2, then take the mean of the similarity for each product 
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    avg_similarity = similarity_matrix.mean(axis=1)
    df1 = df1.copy()
    df1[f"avg_similarity_to_{df2_name}"] = avg_similarity
    return df1

def avg_cosine_similarity_to_multiple(df1_name, df2_names):
    # Calculates the similarity between one target dataset and multiple other datasets
    df1, _ = get_datasets(df1_name, df1_name)
    embeddings1 = np.array(df1["embeddings"].tolist())
    df1_with_similarities = df1.copy()

    for df2_name in df2_names:
        _, df2 = get_datasets(df1_name, df2_name)
        embeddings2 = np.array(df2["embeddings"].tolist())

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        avg_similarity = similarity_matrix.mean(axis=1)
        df1_with_similarities[f"avg_similarity_to_{df2_name}"] = avg_similarity

    return df1_with_similarities

