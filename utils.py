def get_all_domains(path="data/amazon_reviews_categories.txt"):
    with open(path, "r") as file:
        domains = file.read().splitlines()
    return domains


def get_domain_reviews(domain, path="data/reviews"):
    from datasets import load_from_disk

    file_path = f"{path}/{domain}_reviews/"
    return load_from_disk(file_path)


def get_domain_metadata(domain, path="data/metadata"):
    from datasets import load_from_disk

    file_path = f"{path}/{domain}_metadata/"
    return load_from_disk(file_path)
