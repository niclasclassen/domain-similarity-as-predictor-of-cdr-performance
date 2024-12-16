import os
import shutil
import pandas as pd
from datasets import load_dataset, load_from_disk


class DataPreprocessor:
    def __init__(self, output_path):
        self.output_path = output_path

    def preprocess_category(self, reviews_df, metadata_df, category):
        items_with_min_reviews = self._get_items_with_min_reviews(reviews_df)

        
        filtered_metadata_df = metadata_df[metadata_df["parent_asin"].isin(items_with_min_reviews)]

     
        self._save_filtered_metadata(filtered_metadata_df, category)

    def _get_items_with_min_reviews(self, reviews_df):
        review_counts = reviews_df.groupby('parent_asin').size().reset_index(name='review_count')

        # Filter for items with at least 5 reviews
        items_with_min_reviews = review_counts[review_counts['review_count'] >= 5]['parent_asin'].tolist()

        print(f"Found {len(items_with_min_reviews)} items with at least 5 reviews.")
        return items_with_min_reviews

    def _save_filtered_metadata(self, filtered_metadata_df, category):
        category_output_path = os.path.join(self.output_path, category)
        os.makedirs(category_output_path, exist_ok=True)

        # Save the filtered metadata to a CSV file
        filtered_metadata_df.to_csv(f"{category_output_path}/{category}_filtered_metadata.csv", index=False)
        print(f"Saved filtered metadata for category {category}.")


def download_and_process_categories(categories, dataset, raw_review_data_suffix, raw_metadata_suffix, data_path, output_path):
    preprocessor = DataPreprocessor(output_path)

    for category in categories:
        print(f"Processing category: {category}")

        
        print(f"Downloading data for category: {category}")
        dataset_reviews_name = raw_review_data_suffix + category
        dataset_metadata_name = raw_metadata_suffix + category

       
        reviews = load_dataset(dataset, dataset_reviews_name, trust_remote_code=True)
        reviews_path = f"{data_path}/reviews/{category}_reviews"
        reviews.save_to_disk(reviews_path)

        metadata = load_dataset(dataset, dataset_metadata_name, trust_remote_code=True)
        metadata_path = f"{data_path}/metadata/{category}_metadata"
        metadata.save_to_disk(metadata_path)

        reviews_df = reviews["full"].to_pandas()
        metadata_df = metadata["full"].to_pandas()

        preprocessor.preprocess_category(reviews_df, metadata_df, category)

        print(f"Cleaning up metadata for category: {category}")
        shutil.rmtree(metadata_path)
        print(f"Finished processing category: {category}")


# Configuration
dataset = "McAuley-Lab/Amazon-Reviews-2023"
raw_review_data_suffix = "raw_review_"
raw_metadata_suffix = "raw_meta_"
path_to_amazon_reviews_categories = "data/amazon_reviews_categories.txt"
data_path = "data"
output_path = "preprocessed_data"

# Read categories from file
with open(path_to_amazon_reviews_categories, "r") as f:
    categories = f.read().splitlines()

# Process categories
download_and_process_categories(categories, dataset, raw_review_data_suffix, raw_metadata_suffix, data_path, output_path)

