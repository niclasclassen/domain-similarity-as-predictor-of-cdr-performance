from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import json 

def load_data(dataset_name, split_type):
    review = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{dataset_name}", split=split_type)
    meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{dataset_name}", split=split_type)
    return review, meta

def load_data_from_file(file_name):
    with open("Data/" + file_name, "r") as file:
        dataset = json.load(file)
    return dataset

def save_data(dataset, file_name):
    dataset.save_to_disk(f"Data/{file_name}")

def open_dataset(dir_name):
    dataset = load_from_disk(f"Data\{dir_name}")
    return dataset

def merge_columns(dataset): 
    dataset = dataset.map(lambda x: {
        "combined": x["title"] + " " + " ".join(x["features"]) + " " + " ".join(x["description"])
    })
    return dataset

def generate_embeddings(dataset):
    # Load the model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # Generate embeddings for the "combined" column
    embeddings = model.encode(dataset["combined"], batch_size=32, show_progress_bar=True)
    # Use the map function to add embeddings to each row
    def add_embeddings(example, idx):
        example["embeddings"] = embeddings[idx]
        return example
    # Apply the add_embeddings function to each row
    dataset = dataset.map(add_embeddings, with_indices=True)
    
    return dataset

def generate_combined_column(dataset):
    dataset = dataset.map(lambda x: {
        "combined": x["title"] + " " + " ".join(x["features"]) + " " + " ".join(x["description"])
    })
    return dataset
def main(dataset_name):
    datasets = dataset_name
    for i in datasets: 
        review, meta = load_data(i, "full[:10%]")
        save_data(review, i + "_review")
        save_data(meta, i + "_meta")



