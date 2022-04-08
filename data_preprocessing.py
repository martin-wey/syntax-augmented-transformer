import os
import argparse
import logging
import sys
import pickle

import networkx as nx
from tree_sitter import Parser
from datasets import load_dataset, concatenate_datasets

from collators import convert_sample_to_features
from data.code2ast import code2ast, has_error
from data.utils import LANGUAGE_GRAMMARS, preprocess_code
from utils import get_non_terminals_labels

logger = logging.getLogger(__name__)


def filter_sample(code, lang, parser):
    try:
        code = preprocess_code(code, lang)
        G = code2ast(code=code, parser=parser)
        assert nx.is_tree(nx.Graph(G))
        assert nx.is_connected(nx.Graph(G))
    except:
        return False
    if has_error(G):
        return False
    return True


def generate_datasets(lang, args):
    parser = Parser()
    parser.set_language(LANGUAGE_GRAMMARS[lang])

    logger.info('Loading dataset.')
    dataset_path = os.path.join(args.dataset_dir, lang)
    data_files = {
        'train': os.path.join(dataset_path, 'train.jsonl'),
        'valid': os.path.join(dataset_path, 'valid.jsonl'),
        'codebase': os.path.join(dataset_path, 'codebase.jsonl'),
        'test': os.path.join(dataset_path, 'test.jsonl')
    }
    dataset = {}
    for key, item in data_files.items():
        dataset[key] = load_dataset('json', data_files=data_files, split=key)

    # filter codes that cannot be parsed with tree_sitter
    logger.info('Filtering datasets.')
    for key, item in dataset.items():
        dataset[key] = dataset[key].filter(lambda e: filter_sample(e['original_string'], lang, parser), num_proc=8)
        dataset[key] = dataset[key].map(lambda e: {'original_string': preprocess_code(e['original_string'], lang)},
                                        num_proc=8)
        dataset[key] = dataset[key].map(lambda e: convert_sample_to_features(e['original_string'], parser), num_proc=8)
        # dataset[key].save_to_disk(f'{dataset_path}/{key}')

    # get class labels-ids mapping for c and u
    labels_file_path = os.path.join(dataset_path, 'dcu_labels.pkl')
    # convert each non-terminal labels to its id
    labels_to_ids_c = get_non_terminals_labels((dataset['train']['c'], dataset['valid']['c'],
                                                dataset['codebase']['c'], dataset['test']['c']))
    # ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
    labels_to_ids_u = get_non_terminals_labels((dataset['train']['u'], dataset['valid']['u'],
                                                dataset['codebase']['u'], dataset['test']['u']))
    # ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
    return dataset, labels_to_ids_c, labels_to_ids_u


def store_vocabs(path, labels_to_ids_c, labels_to_ids_u):
    ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
    ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
    with open(path, 'wb') as f:
        pickle.dump({
            'labels_to_ids_c': labels_to_ids_c, 'ids_to_labels_c': ids_to_labels_c,
            'labels_to_ids_u': labels_to_ids_u, 'ids_to_labels_u': ids_to_labels_u
        }, f)


def store_datasets(path, dataset):
    for key, item in dataset.items():
        dataset[key].save_to_disk(f'{path}/{key}')


def merge_vocabs(l2id1, l2id2):
    result = dict(l2id1)
    for x, y in l2id2.items():
        if x in result:
            continue
        else:
            new_id = len(result)
            result[x] = new_id
    return result


def merge_datasets(d1, d2):
    result = {}
    for key, item in d1.items():
        result[key] = concatenate_datasets(d1[key], d2[key])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='Root directory of the dataset.')
    parser.add_argument('--lang', type=str, default='python')
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level="INFO"
    )

    if args.lang == 'all':
        dataset, labels_to_ids_c, labels_to_ids_u = {}, {}, {}
        for lang in ['python', 'javascript', 'go', 'java']:
            dataset_temp, labels_to_ids_c_temp, labels_to_ids_u_temp = generate_datasets(lang, args)
            dataset = merge_datasets(dataset, dataset_temp)
            labels_to_ids_c = merge_vocabs(labels_to_ids_c, labels_to_ids_c_temp)
            labels_to_ids_u = merge_vocabs(labels_to_ids_u, labels_to_ids_u_temp)
        dataset_path = os.path.join(args.dataset_dir, 'all')
        store_datasets(dataset_path, dataset)
        labels_file_path = os.path.join(dataset_path, 'dcu_labels.pkl')
        store_vocabs(labels_file_path, labels_to_ids_c, labels_to_ids_u)
    else:
        dataset, labels_to_ids_c, labels_to_ids_u = generate_datasets(args.lang, args)
        dataset_path = os.path.join(args.dataset_dir, args.lang)
        store_datasets(dataset_path, dataset)
        labels_file_path = os.path.join(dataset_path, 'dcu_labels.pkl')
        store_vocabs(labels_file_path, labels_to_ids_c, labels_to_ids_u)