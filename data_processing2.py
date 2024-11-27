import os
import pandas as pd
from datasets import load_from_disk

class DataPreprocessor:
    def __init__(self, path_to_reviews_data, path_to_metadata_data, output_path):
        self.path_to_reviews_data = path_to_reviews_data
        self.path_to_metadata_data = path_to_metadata_data
        self.output_path = output_path

    def preprocess(self, categories):
        for category in categories:
            print(f"Processing category: {category}")
            self.preprocess_category(category)

    def preprocess_category(self, category):
        print(f"Inspecting metadata dataset for category {category}...")

        # Load the reviews and metadata datasets
        reviews = load_from_disk(f"{self.path_to_reviews_data}/{category}_reviews")
        metadata = load_from_disk(f"{self.path_to_metadata_data}/{category}_metadata")

        # Convert datasets to pandas DataFrames
        reviews_df = reviews["full"].to_pandas()
        metadata_df = metadata["full"].to_pandas()

        # Get items with at least 5 reviews from the reviews dataset
        items_with_min_reviews = self._get_items_with_min_reviews(reviews_df)

        # Filter metadata based on items with at least 5 reviews
        filtered_metadata_df = metadata_df[metadata_df["parent_asin"].isin(items_with_min_reviews)]

        # Save the filtered metadata
        self._save_filtered_metadata(filtered_metadata_df, category)

    def _get_items_with_min_reviews(self, reviews_df):
        # Count reviews per parent_asin
        review_counts = reviews_df.groupby('parent_asin').size().reset_index(name='review_count')

        # Filter for items with at least 5 reviews
        items_with_min_reviews = review_counts[review_counts['review_count'] >= 5]['parent_asin'].tolist()

        print(f"Found {len(items_with_min_reviews)} items with at least 5 reviews.")
        return items_with_min_reviews

    def _save_filtered_metadata(self, filtered_metadata_df, category):
        # Ensure output folder exists
        category_output_path = os.path.join(self.output_path, category)
        os.makedirs(category_output_path, exist_ok=True)

        # Save the filtered metadata to a CSV file
        filtered_metadata_df.to_csv(f"{category_output_path}/{category}_filtered_metadata.csv", index=False)
        print(f"Saved filtered metadata for category {category}.")

# Example usage 
path_to_reviews_data = "data/reviews"
path_to_metadata_data = "data/metadata" 
output_path = "preprocessed_data"

categories = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive"]

preprocessor = DataPreprocessor(path_to_reviews_data, path_to_metadata_data, output_path)
preprocessor.preprocess(categories)

