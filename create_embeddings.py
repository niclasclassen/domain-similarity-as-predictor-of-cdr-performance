from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import json 
import pandas as pd 

class PreProcess_embeddings:
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def sample_dataset(self, dataset):
        return dataset.sample(n=100000, random_state=1).reset_index(drop=True)
    def load_data(self, dataset_name): 
        dataset_path = f"preprocessed_data/{dataset_name}/{dataset_name}_filtered_metadata.csv"
        dataset = pd.read_csv(dataset_path)
        # remove the images, videos, store and author columns
        dataset = dataset.drop(columns=["store", "author"])
        return dataset
    
    def combine_columns(self, dataset):
        dataset["combined"] = dataset["title"] + " " + dataset["features"] + " " + dataset["description"]
        return dataset
    
    def generate_embeddings(self, dataset):
        dataset["combined"] = dataset["combined"].fillna("").astype(str)
        # Generate embeddings for the "combined" column
        embeddings = self.model.encode(dataset["combined"].tolist(), batch_size=32, show_progress_bar=True)
        # Ensure the embeddings are added to the DataFrame as a list of lists
        dataset["embeddings"] = list(embeddings)
        
        return dataset
    
    def save_embeddings(self, dataset, dataset_name):
        dataset.to_csv(f"preprocessed_data/{dataset_name}/{dataset_name}_embeddings.csv", index=False)

    def main(self):
        for dataset_name in self.dataset_list:
            dataset = self.load_data(dataset_name)
            print(f"Processing dataset: {dataset_name}")
            #dataset = self.sample_dataset(dataset)
            dataset = self.combine_columns(dataset)
            dataset = self.generate_embeddings(dataset)
            self.save_embeddings(dataset, dataset_name)

if __name__ == "__main__":
    path_to_amazon_reviews_categories = "data/amazon_reviews_categories.txt"
    
    with open(path_to_amazon_reviews_categories, "r") as f:
        dataset_list = f.read().splitlines()

    preprocessor = PreProcess_embeddings(dataset_list)
    preprocessor.main()