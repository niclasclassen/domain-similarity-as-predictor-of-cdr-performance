from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from tqdm import tqdm
import random

def get_datasets(dataset_name1, dataset_name2):
    """
    Function to read the data used for similarity the similarity calculations.
    takes in two csv files with an "embeddings" column and returns the dataframes.
    """
    df1 = pd.read_csv(f"preprocessed_data\\{dataset_name1}\\{dataset_name1}_embeddings.csv")
    df2 = pd.read_csv(f"preprocessed_data\\{dataset_name2}\\{dataset_name2}_embeddings.csv")

    return df1, df2

def cosine_similarity_pairs():
    """
    Calculates the cosine similarity between the embeddings of specific dataset pairs at the item level.
    creates a csv file with the similarity values for each item in the first dataset to each item in the second dataset.
    """
    # dataset_pairs = [("Musical_Instruments", "Software"), 
    #                  ("Software", "Video_Games"), 
    #                  ("Musical_Instruments", "Video_Games"),
    #                    ("Kindle_Store","Books"), ("CDs_and_Vinyl","Books"), ("Electronics","Books")]
    dataset_pairs = [("Health_and_Household","Books")]
    chunk_size = 500  # Define chunk size for embeddings
    max_items = 250000  # Define the maximum number of items to process

    for df1_name, df2_name in tqdm(dataset_pairs):
        print(f"\n----Processing pairs {df1_name} and {df2_name}----")  # Debugging
        
        # Retrieve the datasets
        df1, df2 = get_datasets(df1_name, df2_name)
        print("finished reading datasets")  # Debugging
        df1 = sample_dataset(df1, max_items)
        df2 = sample_dataset(df2, max_items)
        print("sampled datasets")  # Debugging
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
        print("Embeddings normalized")  # Debugging
        # Quantize embeddings to int8
        def quantize_to_int8(df):
            df["embeddings"] = df["embeddings"].apply(
                lambda x: (x * 127).astype(np.int8)
            )
            return df

        df1 = quantize_to_int8(df1)
        df2 = quantize_to_int8(df2)
        print("Embeddings quantized")  # Debugging

        embeddings1 = np.array(df1["embeddings"].tolist(), dtype=np.int8)
        embeddings2 = np.array(df2["embeddings"].tolist(), dtype=np.int8)

        print("Quantized embeddings converted to NumPy arrays")  # Debugging

        # Compute similarity in chunks
        num_rows_df1 = embeddings1.shape[0]
        num_rows_df2 = embeddings2.shape[0]
        avg_similarities = np.zeros(len(embeddings1))
        
        
        with tqdm(total=num_rows_df1 * num_rows_df2, desc="Processing chunks", leave=False) as pbar:
            # Initialize an array to hold the averaged similarities for df1
            # Process embeddings in chunks
            for start_idx1 in tqdm(range(0, embeddings1.shape[0], chunk_size), desc="Processing df1 chunks"):
                chunk1 = embeddings1[start_idx1:start_idx1 + chunk_size]
                chunk1_rescaled = chunk1.astype(np.float32) / 127 * max_val1

                # Temporary storage for this chunk's averaged similarities
                chunk_avg_similarities = np.zeros(chunk1.shape[0])

                for start_idx2 in range(0, embeddings2.shape[0], chunk_size):
                    chunk2 = embeddings2[start_idx2:start_idx2 + chunk_size]
                    chunk2_rescaled = chunk2.astype(np.float32) / 127 * max_val2

                    # Compute cosine similarity for the current chunk pair
                    similarity_matrix = cosine_similarity(chunk1_rescaled, chunk2_rescaled)

                    # Accumulate similarity sums for averaging
                    chunk_avg_similarities += similarity_matrix.mean(axis=1)

                # Divide by the number of chunks in df2 to get the final average
                num_chunks_df2 = -(-embeddings2.shape[0] // chunk_size)  # Ceiling division
                chunk_avg_similarities /= num_chunks_df2

                # Store the chunk's averaged similarities into the final array
                avg_similarities[start_idx1:start_idx1 + chunk1.shape[0]] = chunk_avg_similarities

            # Add the calculated similarities to df1
            df1[f"avg_similarity_to_{df2_name}"] = avg_similarities


        print(f"Added similarity column for {df1_name} to {df2_name}")  # Debugging

        # Save the updated dataframe
        df1.to_csv(f"results/{df1_name}_to_{df2_name}_similarities.csv", index=False)

    print("Finished processing all specified pairs.")  # Debugging

########################################

def sample_dataset(df, n):
    if len(df) > n:
        sampled_df = df.sample(n=n, random_state=42).reset_index(drop=True)
    else:
        sampled_df = df  # If the dataset is smaller than n, use the whole dataset
    return sampled_df

def cosine_similarity_matrix_final():
    path_to_amazon_reviews_categories = "data/amazon_reviews_categories.txt"
    with open(path_to_amazon_reviews_categories, "r") as f:
        df_names = f.read().splitlines()

    # Calculates a similarity matrix for unique dataset combinations
    similarity_results = []
    dataset_combinations = combinations_with_replacement(df_names, 2)

    chunk_size = 500  # Define chunk size for embeddings
    max_items = 10000
    valid_domains = ["Books", "Clothing_Shoes_and_Jewelry"]
    #store the df1
    dataframe = None
    previous_dataset_name = ""
    for df1_name, df2_name in tqdm(dataset_combinations):
        if not df1_name in valid_domains:
            break
        print(f"\n----Processing {df1_name} and {df2_name}----")  # Debugging
        if previous_dataset_name != df1_name:
            df1 = get_datasets(df1_name)
            previous_dataset_name = df1_name
            dataframe = df1
        else:
            df1 = dataframe
        # Retrieve the datasets
        print(f"finished reading {df1_name}")
        df2 = get_datasets(df2_name)
        print(f"finished reading {df2_name}")

        df1 = sample_dataset(df1, max_items)
        df2 = sample_dataset(df2, max_items)
        print(f"Sampled datasets to a maximum of {max_items} items\n")
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
        print("Embeddings normalized and quantized")  # Debugging
        embeddings1 = np.array(df1["embeddings"].tolist(), dtype=np.int8)
        embeddings2 = np.array(df2["embeddings"].tolist(), dtype=np.int8)

        print("Quantized embeddings converted to NumPy arrays")  # Debugging

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
        with open("results/results_sampled.txt", "a") as f:
            f.write(f"{df1_name} \t {df2_name} \t {avg_similarity_full}\n")

    print("Finished processing, saving last items.")  # Debugging

    # Convert to DataFrame for better visualization
    similarity_results_df = pd.DataFrame(similarity_results)

    # Save the results to a file
    try:
        print("Saving full similarity results to a file")  # Debugging
        similarity_results_df.to_csv("results/similarity_results_sampled.csv", index=False)
    except Exception as e:
        print(f"Could not save the results to a file: {e}")

    return similarity_results_df

def process_dataset_pairs(file_path):
    """
    takes the results from the similarity calculations and returns the top 2, middle 1 and least 2 dataset pairs.
    """
    columns = ["dataset1", "dataset2", "similarity"]
    df = pd.read_csv(file_path, sep="\t", names=columns)
    df["dataset1"] = df["dataset1"].str.strip()
    df["dataset2"] = df["dataset2"].str.strip()
    df = df[df["dataset1"] != df["dataset2"]]

    
    df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
    top_2 = df.iloc[:2]  
    middle_1 = df.iloc[[len(df) // 2]] 
    least_2 = df.iloc[-2:]  

    result = pd.concat([top_2, middle_1, least_2])
    dataset_pairs = list(zip(result["dataset1"], result["dataset2"]))

    return dataset_pairs


def main():
    # Calculate the average similarity between two full datasets
    # similarity_results_df = cosine_similarity_matrix_final()
    # print("finished caluculating similarity")
    # print(similarity_results_df)

    # Calculate the similarity between pairs of datasets at the item level
    #dataset_pairs = process_dataset_pairs("results/results_sampled.txt") # get the dataset pairs
    cosine_similarity_pairs() # calculate the similarity between the pairs

if __name__ == "__main__":
    main()
