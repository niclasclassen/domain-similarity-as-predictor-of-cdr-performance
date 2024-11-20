def download_reviews(
    dataset="McAuley-Lab/Amazon-Reviews-2023",
    raw_review_data_suffix="raw_review_",
    raw_metadata_suffix="raw_meta_",
    path_to_amazon_reviews_categories="data/amazon_reviews_categories.txt",
    data_path="data",
):
    from datasets import load_dataset

    # read categories from file
    with open(path_to_amazon_reviews_categories, "r") as f:
        categories = f.read().splitlines()

    for category in categories:
        print(f"Downloading reviews for category: {category}")
        dataset_reviews_name = raw_review_data_suffix + category
        dataset_metadata_name = raw_metadata_suffix + category

        category_reviews = load_dataset(
            dataset, dataset_reviews_name, trust_remote_code=True
        )
        category_metadata = load_dataset(
            dataset, dataset_metadata_name, trust_remote_code=True
        )

        category_reviews.save_to_disk(f"{data_path}/reviews/{category}_reviews")
        category_metadata.save_to_disk(f"{data_path}/metadata/{category}_metadata")


def prepare_source_domain_data():
    pass


def create_item_embeddings(
    path_to_reviews_data="data/reviews",
    path_to_metadata_data="data/metadata",
    download_reviews=False,
):
    from datasets import load_from_disk

    if download_reviews:
        download_reviews()

    # read categories from file
    with open("data/amazon_reviews_categories.txt", "r") as f:
        categories = f.read().splitlines()

    for category in categories:
        print(f"Creating item embeddings for category: {category}")
        reviews = load_from_disk(f"{path_to_reviews_data}/{category}_reviews")
        metadata = load_from_disk(f"{path_to_metadata_data}/{category}_metadata")


if __name__ == "__main__":
    download_reviews()
