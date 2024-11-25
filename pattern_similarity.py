from utils import get_all_domains, get_domain_reviews


def get_rating_pattern_similarities(path="data/reviews"):
    print("Hi")
    domains = get_all_domains()
    for domain in domains:
        print(f"Processing domain: {domain}")
        raw_reviews = get_domain_reviews(domain, path)
        reviews_df = process_reviews(raw_reviews)

        # create user-item interaction matrix
        print("Creating user-item interaction matrix")
        user_item_interaction_matrix = create_user_item_interaction_matrix(reviews_df)

        # compute pattern similarities
        print("Computing pattern similarities")
        item_rating_similarities = compute_item_rating_similarities(
            user_item_interaction_matrix
        )
        print("Computing pattern similarities")
        user_rating_similarities = compute_user_rating_similarities(
            user_item_interaction_matrix
        )


def process_reviews(reviews):
    import pandas as pd

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(reviews["full"])

    # Only keep user_id, product_id, and rating columns
    df = df[["user_id", "parent_asin", "rating", "timestamp"]]

    # only keep latest 1000 reviews
    df = df.sort_values("timestamp", ascending=False).head(1000)
    df = df.drop(columns=["timestamp"])

    return df


def create_user_item_interaction_matrix(reviews_df):
    import pandas as pd

    # Create a user-item interaction matrix
    user_item_matrix = reviews_df.pivot(
        index="user_id", columns="parent_asin", values="rating"
    )

    # Fill empty values with zeros
    user_item_matrix = user_item_matrix.fillna(0)

    return user_item_matrix


def compute_item_rating_similarities(reviews_df):
    # Embed the items via their ratings
    # Calculate the count of each rating for each item
    rating_counts = (
        reviews_df.groupby(["item_id", "rating"]).size().unstack(fill_value=0)
    )

    # Normalize the counts to get the proportion of each rating
    rating_proportions = rating_counts.div(rating_counts.sum(axis=1), axis=0)

    # Fill missing values with zeros (in case some items don't have all ratings)
    rating_proportions = rating_proportions.fillna(0)

    print(rating_proportions)

    # Display the rating proportions


def compute_user_rating_similarities(user_item_interaction_matrix):
    from sklearn.metrics.pairwise import cosine_similarity

    item_similarity = cosine_similarity(user_item_interaction_matrix.T)


if __name__ == "__main__":
    get_rating_pattern_similarities()
