import pandas as pd
import time
import gzip
import json

def filter_items(dataset_name):
    # File paths
    csv_path = f"preprocessed_data/{dataset_name}/{dataset_name}_embeddings.csv"
    jsonl_path = f"data/reviews/{dataset_name}_reviews/{dataset_name}.jsonl.gz"

    # Load the items dataset
    items_df = pd.read_csv(csv_path)

    # Get the current timestamp and calculate the timestamp for 3 years ago
    current_time = int(time.time())
    three_years_ago = current_time - 3 * 365 * 24 * 60 * 60  # Approximation of 3 years in seconds

    # Initialize a list to store filtered reviews
    filtered_reviews = []

    # Read the reviews dataset in chunks and filter by parent_asin and timestamp
    with gzip.open(jsonl_path, 'rt') as file:
        for line in file:
            try:
                record = json.loads(line)
                # Filter reviews: check if the parent_asin exists in items_df and if the review is from the last 3 years
                if record["parent_asin"] in items_df["parent_asin"].values and record["timestamp"] >= three_years_ago:
                    filtered_reviews.append(record)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")

    # Convert the filtered reviews into a DataFrame
    reviews_df = pd.DataFrame(filtered_reviews)

    # Count reviews per parent_asin
    review_counts = reviews_df['parent_asin'].value_counts()

    # Filter parent_asins with more than 5 reviews
    valid_asins = review_counts[review_counts > 5].index

    # Filter the items DataFrame to only include rows with valid parent_asins
    filtered_items = items_df[items_df['parent_asin'].isin(valid_asins)]

    # Save the filtered data to a new CSV file
    filtered_items.to_csv(f"preprocessed_data/{dataset_name}/{dataset_name}_final.csv", index=False)

    # Return the filtered DataFrame
    return filtered_items

# Example usage:
# filtered_items = filter_items("Automotive")
