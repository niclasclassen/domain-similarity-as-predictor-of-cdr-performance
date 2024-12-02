# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse

from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CMF', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--source_domain', type=str, default='', help='source domain')
    parser.add_argument('--target_domain', type=str, default='', help='target domain')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    results = run_recbole_cdr(model=args.model, config_file_list=config_file_list)
    # save the results to a file
    with open(f'./results/{args.model}_results.txt', 'w') as f:
        f.write(str(f"Source Domain: {args.source_domain}, Target Domain: {args.target_domain}\n {results}"))
