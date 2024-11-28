def get_all_domains(path="data/amazon_reviews_categories.txt"):
    with open(path, "r") as file:
        domains = file.read().splitlines()
    return domains


def get_domain_reviews(domain, path="data/reviews/json"):
    import pandas as pd
    import json

    file_path = f"{path}/{domain}_reviews"

    # Read the JSONL file
    reviews = []
    with open(file_path, "r") as file:
        for line in file:
            reviews.append(json.loads(line))

    # Convert to pandas DataFrame
    df = pd.DataFrame(reviews)

    return df

    # from datasets import load_from_disk

    # file_path = f"{path}/{domain}_reviews/"
    # return load_from_disk(file_path)


def get_domain_metadata(domain, path="data/metadata"):
    from datasets import load_from_disk

    file_path = f"{path}/{domain}_metadata/"
    return load_from_disk(file_path)
