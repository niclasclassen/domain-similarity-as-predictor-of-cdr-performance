from utils import get_domain_reviews
import numpy as np
import pandas as pd
import argparse
import json
import random
def get_latest_timestamp(domain):
    file_path = f"data/reviews/json/{domain}_reviews"
    latest_review_timestamp = 0
    with open(file_path, "r") as file:
        for line in file:
            review = json.loads(line)
            latest_review_timestamp = max(latest_review_timestamp, review["timestamp"])
    return latest_review_timestamp
def fix_niclas_mistake(domain): 
    path = f"results/full_split_cdr_data/{domain}_reviews"
    reviews = pd.read_csv(path, sep='\t')
    print("read data")
    reviews = reviews[['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']]
    reviews.to_csv(f'results/full_split_cdr_data/{domain}_reviews', sep='\t', index=False)
    
def sample_data(domain):
    path = f"results/full_split_cdr_data/{domain}_reviews"
    reviews = pd.read_csv(path, sep='\t')
    print("read data")
    reviews = reviews.sample(6000000, random_state=42).reset_index(drop=True)
    reviews.to_csv(f'results/full_split_cdr_data/{domain}_reviews_sampled', sep='\t', index=False)
    print("saved data")

def get_domain_reviews_data(domain):
    print("getting latest timestamp")
    latest_timestamp = get_latest_timestamp(domain)
    print("got latest timestamp")
    file_path = f"data/reviews/json/{domain}_reviews"
    # only keep the reviews from the last 5 years 
    five_years_ago = latest_timestamp - 5 * 365 * 24 * 60 * 60 * 1000
    reviews_list = []
    with open(file_path, "r") as file:
        for line in file:
            review = json.loads(line)
            if review["timestamp"] >= five_years_ago:
                filtered_review = {key: review[key] for key in ["user_id", "parent_asin", "rating", "timestamp"]}
                reviews_list.append(filtered_review)
    
    reviews = pd.DataFrame(reviews_list)
    print("converted to dataframe")
    reviews = reviews[['user_id', 'parent_asin', 'rating', 'timestamp']]
    reviews = reviews[['user_id', 'parent_asin', 'rating', 'timestamp']].rename(columns={
    'user_id': 'user_id:token',
    'parent_asin': 'item_id:token',
    'rating': 'rating:float',
    'timestamp': 'timestamp:float'
    })
    reviews.to_csv(f'results/full_split_cdr_data/{domain}_reviews', sep='\t', index=False)
    print("saved to csv")

def make_rec_df(source_domain, target_domain):
    source_domain = get_domain_reviews(source_domain)
    target_domain = get_domain_reviews(target_domain)

    source_domain = source_domain[['user_id', 'parent_asin', 'rating', 'timestamp']]
    source_domain = source_domain[['user_id', 'parent_asin', 'rating', 'timestamp']].rename(columns={
    'user_id': 'user_id:token',
    'parent_asin': 'item_id:token',
    'rating': 'rating:float',
    'timestamp': 'timestamp:float'
    })
    target_domain = target_domain[['user_id', 'parent_asin', 'rating', 'timestamp']]
    target_domain = target_domain[['user_id', 'parent_asin', 'rating', 'timestamp']].rename(columns={
    'user_id': 'user_id:token',
    'parent_asin': 'item_id:token',
    'rating': 'rating:float',
    'timestamp': 'timestamp:float'
    })

    latest_review_source = source_domain['timestamp:float'].max()
    five_years_ago = latest_review_source - 5 * 365 * 24 * 60 * 60 * 1000

    # source_domain = source_domain[source_domain['timestamp:float'] >= five_years_ago]
    n = 250000
    source_domain = source_domain.sample(n, random_state=42).reset_index(drop=True)

    latest_review_target = target_domain['timestamp:float'].max()
    five_years_ago = latest_review_target - 5 * 365 * 24 * 60 * 60 * 1000

    target_domain = target_domain[target_domain['timestamp:float'] >= five_years_ago]
    target_domain = target_domain.sample(n, random_state=42).reset_index(drop=True)
    
    target_domain.to_csv('recbole_cdr/dataset/target_domain_reviews/target_domain_reviews.inter', sep='\t', index=False)
    source_domain.to_csv('recbole_cdr/dataset/source_domain_reviews/source_domain_reviews.inter', sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_domain', type=str, default='', help='source domain')
    # parser.add_argument('--target_domain', type=str, default='', help='target domain')
    parser.add_argument("--domain", type=str, default='all', help="domain to get reviews from")
    args = parser.parse_args()
    # get_domain_reviews_data(args.domain)
    # fix_niclas_mistake(args.domain)
    sample_data(args.domain)
    # make_rec_df(args.source_domain, args.target_domain)
    # print('Done')

if __name__ == '__main__':
    main()
