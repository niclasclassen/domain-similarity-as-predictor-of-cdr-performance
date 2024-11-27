from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
from datasets import load_dataset, load_from_disk
import Data_processing_ as dp
from joblib import Parallel, delayed
from itertools import combinations_with_replacement
from tqdm import tqdm
def get_datasets(dataset_name1, dataset_name2):
    df1 = pd.read_csv(f"preprocessed_data\\{dataset_name1}\\{dataset_name1}_embeddings.csv")
    df2 = pd.read_csv(f"preprocessed_data\\{dataset_name2}\\{dataset_name2}_embeddings.csv")

    return df1, df2

def get_dataset(dataset_name1):
    dataset1 = load_from_disk(f"Data\\metadata\\{dataset_name1}_metadata\\full")

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
########################################

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations_with_replacement
from tqdm import tqdm


def cosine_similarity_matrix_final():
    path_to_amazon_reviews_categories = "data/amazon_reviews_categories.txt"
    with open(path_to_amazon_reviews_categories, "r") as f:
        df_names = f.read().splitlines()

    # Calculates a similarity matrix for unique dataset combinations
    similarity_results = []
    dataset_combinations = combinations_with_replacement(df_names, 2)

    chunk_size = 500  # Define chunk size for embeddings

    for df1_name, df2_name in tqdm(dataset_combinations):
        print(f"\nProcessing {df1_name} and {df2_name}\n")  # Debugging

        # Retrieve the datasets
        df1, df2 = get_datasets(df1_name, df2_name)

        # Normalize embeddings
        def normalize_embeddings(df):
            df["embeddings"] = df["embeddings"].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ')
            )
            max_val = np.abs(df["embeddings"].apply(np.max).max())
            df["embeddings"] = df["embeddings"].apply(lambda x: x / max_val)
            return max_val, df

        max_val1, df1 = normalize_embeddings(df1)
        max_val2, df2 = normalize_embeddings(df2)

        # Quantize embeddings to int8
        def quantize_to_int8(df):
            df["embeddings"] = df["embeddings"].apply(
                lambda x: (x * 127).astype(np.int8)
            )
            return df

        df1 = quantize_to_int8(df1)
        df2 = quantize_to_int8(df2)
        print("Embeddings normalized and quantized\n")  # Debugging
        embeddings1 = np.array(df1["embeddings"].tolist(), dtype=np.int8)
        embeddings2 = np.array(df2["embeddings"].tolist(), dtype=np.int8)

        print("Quantized embeddings converted to NumPy arrays\n")  # Debugging

        # Compute similarity in chunks
        num_rows_df1 = embeddings1.shape[0]
        num_rows_df2 = embeddings2.shape[0]

        avg_similarity_full = 0
        count = 0
        with tqdm(total=num_rows_df1 * num_rows_df2, desc="Processing chunks", leave=False) as pbar:
            for start_idx1 in range(0, num_rows_df1, chunk_size):
                chunk1 = embeddings1[start_idx1:start_idx1 + chunk_size]

                for start_idx2 in range(0, num_rows_df2, chunk_size):
                    chunk2 = embeddings2[start_idx2:start_idx2 + chunk_size]

                    # Rescale chunks
                    chunk1_rescaled = chunk1.astype(np.float32) / 127 * max_val1
                    chunk2_rescaled = chunk2.astype(np.float32) / 127 * max_val2

                    # Compute cosine similarity
                    similarity_matrix = cosine_similarity(chunk1_rescaled, chunk2_rescaled)

                    # Compute average similarity for the current chunk
                    avg_similarity_full += similarity_matrix.sum()
                    count += similarity_matrix.size

        avg_similarity_full /= count

        print("Cosine similarity computed, saving results")  # Debugging

        similarity_results.append({
            "Dataset 1": df1_name,
            "Dataset 2": df2_name,
            "Average Similarity": avg_similarity_full
        })

        # Save results incrementally
        with open("results/results.txt", "a") as f:
            f.write(f"{df1_name} \t {df2_name} \t {avg_similarity_full}\n")

    print("Finished processing, saving last items.\n")  # Debugging

    # Convert to DataFrame for better visualization
    similarity_results_df = pd.DataFrame(similarity_results)

    # Save the results to a file
    try:
        print("Saving full similarity results to a file")  # Debugging
        similarity_results_df.to_csv("results/similarity_results.csv", index=False)
    except Exception as e:
        print(f"Could not save the results to a file: {e}")

    return similarity_results_df



####################
# Chunked Version

def calculate_avg_cosine_similarity_chunked(embeddings1, embeddings2, chunk_size=1000):
    """
    Computes the average cosine similarity between two large embedding matrices
    by processing them in smaller chunks.
    """
    n1, n2 = len(embeddings1), len(embeddings2)
    avg_similarity = 0
    total_count = 0

    # Loop through chunks of embeddings1
    for i in range(0, n1, chunk_size):
        chunk1 = embeddings1[i:i+chunk_size]
        
        # Loop through chunks of embeddings2
        for j in range(0, n2, chunk_size):
            chunk2 = embeddings2[j:j+chunk_size]
            
            # Compute cosine similarity for the current chunk pair
            similarity_matrix = cosine_similarity(chunk1, chunk2)
            avg_similarity += similarity_matrix.sum()
            total_count += similarity_matrix.size

    # Calculate average similarity
    avg_similarity /= total_count
    return avg_similarity


def process_combination_chunked(df1_name, df2_name, chunk_size=1000):
    """
    Processes a single dataset combination using chunked cosine similarity.
    """
    df1, df2 = get_datasets(df1_name, df2_name)
    embeddings1 = np.array(df1["embeddings"].tolist())
    embeddings2 = np.array(df2["embeddings"].tolist())
    avg_similarity = calculate_avg_cosine_similarity_chunked(embeddings1, embeddings2, chunk_size)
    return {
        "Dataset 1": df1_name,
        "Dataset 2": df2_name,
        "Average Similarity": avg_similarity
    }


def cosine_similarity_matrix_chunked(df_names, chunk_size=1000):
    """
    Computes the average cosine similarity matrix for all unique dataset combinations,
    processing embeddings in chunks sequentially.
    """
    from itertools import combinations_with_replacement
    
    similarity_results = []
    dataset_combinations = combinations_with_replacement(df_names, 2)

    for df1_name, df2_name in dataset_combinations:
        result = process_combination_chunked(df1_name, df2_name, chunk_size)
        similarity_results.append(result)

    # Convert results to a DataFrame
    similarity_results_df = pd.DataFrame(similarity_results)
    return similarity_results_df

def main():
    # Calculate the similarity matrix
    similarity_results_df = cosine_similarity_matrix_final()
    print("finished caluculating similarity")
    print(similarity_results_df)

if __name__ == "__main__":
    main()
