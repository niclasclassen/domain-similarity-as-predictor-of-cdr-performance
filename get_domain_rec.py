import pandas as pd
import json


def get_domain_reviews(domain, path=""):
    file_path = f"{domain}_reviews"
    latest_review_timestamp = 1694556792711
    five_years_ago = latest_review_timestamp - 5 * 365 * 24 * 60 * 60 * 1000

    # Read the JSONL file and filter reviews
    reviews = []
    with open(file_path, "r") as file:
        for line in file:
            review = json.loads(line)
            if review["timestamp"] >= five_years_ago:
                reviews.append(review)

    # Convert to pandas DataFrame
    df = pd.DataFrame(reviews)
    return df


def get_latest_timestamp(domain):
    file_path = f"{domain}_reviews"
    latest_review_timestamp = 0
    with open(file_path, "r") as file:
        for line in file:
            review = json.loads(line)
            latest_review_timestamp = max(latest_review_timestamp, review["timestamp"])
    return latest_review_timestamp


def make_rec_df():
    # latest_review_date = get_latest_timestamp("movie_and_tv")
    source_domain = get_domain_reviews("movie_and_tv")

    source_domain = source_domain[["user_id", "parent_asin", "rating", "timestamp"]]
    # Rename columns
    source_domain = source_domain.rename(
        columns={
            "user_id": "user_id:token",
            "parent_asin": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float",
        }
    )

    source_domain.to_csv(
        "movies_and_tv.inter",
        sep="\t",
        index=False,
    )


def main():
    make_rec_df()
    print("Done")


if __name__ == "__main__":
    main()


# import pandas as pd
# import json


# def get_domain_reviews(domain, path="data"):
#     file_path = f"{path}/{domain}_reviews"
#     latest_review_timestamp = 1694657549017
#     latest_review_minus_5_years = (
#         latest_review_timestamp - 5 * 365 * 24 * 60 * 60 * 1000
#     )
#     # Read the JSONL file
#     reviews = []
#     latest_review_timestamp = 0
#     with open(file_path, "r") as file:
#         for line in file:
#             # reviews.append(json.loads(line))
#             line = json.loads(line)
#             timestamp = line["timestamp"]
#             if timestamp > latest_review_timestamp:
#                 reviews.append(line)

#     # Convert to pandas DataFrame
#     df = pd.DataFrame(reviews)
#     return df


# def make_rec_df():
#     source_domain = get_domain_reviews("books")
#     # target_domain = get_domain_reviews("")

#     source_domain = source_domain[["user_id", "parent_asin", "rating", "timestamp"]]
#     source_domain = source_domain[
#         ["user_id", "parent_asin", "rating", "timestamp"]
#     ].rename(
#         columns={
#             "user_id": "user_id:token",
#             "parent_asin": "item_id:token",
#             "rating": "rating:float",
#             "timestamp": "timestamp:float",
#         }
#     )
#     # target_domain = target_domain[["user_id", "parent_asin", "rating", "timestamp"]]
#     # target_domain = target_domain[
#     #     ["user_id", "parent_asin", "rating", "timestamp"]
#     # ].rename(
#     #     columns={
#     #         "user_id": "user_id:token",
#     #         "parent_asin": "item_id:token",
#     #         "rating": "rating:float",
#     #         "timestamp": "timestamp:float",
#     #     }
#     # )

#     latest_review_source = source_domain["timestamp:float"].max()
#     five_years_ago = latest_review_source - 5 * 365 * 24 * 60 * 60 * 1000
#     source_domain = source_domain[source_domain["timestamp:float"] >= five_years_ago]

#     # source_domain = source_domain[source_domain['timestamp:float'] >= five_years_ago]
#     # latest_review_target = target_domain["timestamp:float"].max()
#     # five_years_ago = latest_review_target - 5 * 365 * 24 * 60 * 60 * 1000

#     # target_domain = target_domain[target_domain["timestamp:float"] >= five_years_ago]

#     # target_domain.to_csv(
#     #     "recbole_cdr/dataset/target_domain_reviews/target_domain_reviews.inter",
#     #     sep="\t",
#     #     index=False,
#     # )
#     source_domain.to_csv(
#         "source_domain_reviews.inter",
#         sep="\t",
#         index=False,
#     )


# def main():
#     make_rec_df()
#     print("Done")


# if __name__ == "__main__":
#     main()
