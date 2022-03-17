import random
import logging

import torch
import numpy as np
from transformers import AutoTokenizer
from tree_sitter import Parser


def main():
    # select the parser
    parser = Parser()
    parser.set_language('python')

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')



if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    main()
