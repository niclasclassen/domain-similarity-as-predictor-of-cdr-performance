from utils import get_domain_reviews
import numpy as np
import pandas as pd
import argparse

def calculate_sparsity(df):
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    total_possible_interactions = num_users * num_items
    actual_interactions = len(df)
    return 1 - (actual_interactions / total_possible_interactions)

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
    # source_domain = source_domain.sample(n, random_state=42).reset_index(drop=True)

    latest_review_target = target_domain['timestamp:float'].max()
    five_years_ago = latest_review_target - 5 * 365 * 24 * 60 * 60 * 1000

    target_domain = target_domain[target_domain['timestamp:float'] >= five_years_ago]

    # sparsity_df1 = calculate_sparsity(target_domain)
    # sparsity_df2 = calculate_sparsity(source_domain)

    # target_sparsity = max(sparsity_df1, sparsity_df2)
    
    # target_domain = target_domain.sample(n, random_state=42).reset_index(drop=True)
    target_domain.to_csv('recbole_cdr/dataset/target_domain_reviews/target_domain_reviews.inter', sep='\t', index=False)
    source_domain.to_csv('recbole_cdr/dataset/source_domain_reviews/source_domain_reviews.inter', sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', type=str, default='', help='source domain')
    parser.add_argument('--target_domain', type=str, default='', help='target domain')

    args = parser.parse_args()
    make_rec_df(args.source_domain, args.target_domain)
    print('Done')

if __name__ == '__main__':
    main()
