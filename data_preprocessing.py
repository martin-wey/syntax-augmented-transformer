import os
import argparse
import logging
import sys
import pickle

from tree_sitter import Parser
from datasets import load_dataset

from collators import filter_sample, convert_sample_to_features
from data.utils import LANGUAGE_GRAMMARS
from utils import get_non_terminals_labels

logger = logging.getLogger(__name__)


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

    parser = Parser()
    parser.set_language(LANGUAGE_GRAMMARS[args.lang])

    logger.info('Loading dataset.')
    dataset_path = os.path.join(args.dataset_dir, args.lang)
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
        dataset[key] = dataset[key].filter(lambda e: filter_sample(e['original_string'], args.lang, parser), num_proc=8)
        dataset[key] = dataset[key].map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang), num_proc=8)
        dataset[key].save_to_disk(f'{dataset_path}/{key}')

    # get class labels-ids mapping for c and u
    labels_file_path = os.path.join(dataset_path, 'dcu_labels.pkl')
    # convert each non-terminal labels to its id
    labels_to_ids_c = get_non_terminals_labels((dataset['train']['c'], dataset['valid']['c'],
                                                dataset['codebase']['c'], dataset['test']['c']))
    ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
    labels_to_ids_u = get_non_terminals_labels((dataset['train']['u'], dataset['valid']['u'],
                                                dataset['codebase']['u'], dataset['test']['u']))
    ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
    with open(labels_file_path, 'wb') as f:
        pickle.dump({
            'labels_to_ids_c': labels_to_ids_c, 'ids_to_labels_c': ids_to_labels_c,
            'labels_to_ids_u': labels_to_ids_u, 'ids_to_labels_u': ids_to_labels_u
        }, f)
