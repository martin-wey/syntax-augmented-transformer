import os
import random
import logging
import pickle

import torch
import hydra
import omegaconf
import wandb
import numpy as np
from transformers import RobertaTokenizerFast
from datasets import load_from_disk
from tree_sitter import Parser

from models.encoder_decoder import TransformerEncoder, TransformerEncoderSyntax, TransformerDecoder, TransformerEncoderDecoder
from data.utils import LANGUAGE_GRAMMARS
from utils import convert_to_ids
from train import train_baseline, train_syntax_augmented_trans
from test import test_baseline, test_syntax_augmented_trans

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'baseline_trans': (TransformerEncoder, TransformerDecoder, RobertaTokenizerFast),
    'syntax_augmented_trans': (TransformerEncoderSyntax, TransformerDecoder, RobertaTokenizerFast),
}


@hydra.main(config_path='config', config_name='defaults')
def main(cfg: omegaconf.DictConfig):
    if cfg.run.seed > 0:
        random.seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        torch.manual_seed(cfg.run.seed)
        torch.cuda.manual_seed_all(cfg.run.seed)

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.parallel = torch.cuda.device_count() > 1
    cfg.run.base_path = hydra.utils.get_original_cwd()

    if cfg.use_wandb:
        wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(**cfg.wandb.setup, config=wandb_cfg)

    # select the parser
    parser = Parser()
    parser.set_language(LANGUAGE_GRAMMARS[cfg.run.dataset_lang])

    if cfg.run.do_train or cfg.run.do_test:
        logger.info('Loading train/valid datasets.')
        dataset_path = os.path.join(cfg.run.base_path, cfg.run.dataset_dir, cfg.run.dataset_lang)
        train_dataset = load_from_disk(f'{dataset_path}/train')
        valid_dataset = load_from_disk(f'{dataset_path}/valid')
        test_dataset = load_from_disk(f'{dataset_path}/test')

        if cfg.model.config == 'syntax_augmented_trans':
            dcu_labels = pickle.load(open(f'{dataset_path}/dcu_labels.pkl', 'rb'))
            train_dataset = train_dataset.map(lambda e: convert_to_ids(e['c'], 'c', dcu_labels['labels_to_ids_c']), num_proc=4)
            valid_dataset = valid_dataset.map(lambda e: convert_to_ids(e['c'], 'c', dcu_labels['labels_to_ids_c']), num_proc=4)
            test_dataset = test_dataset.map(lambda e: convert_to_ids(e['c'], 'c', dcu_labels['labels_to_ids_c']), num_proc=4)

            train_dataset = train_dataset.map(lambda e: convert_to_ids(e['u'], 'u', dcu_labels['labels_to_ids_u']), num_proc=4)
            valid_dataset = valid_dataset.map(lambda e: convert_to_ids(e['u'], 'u', dcu_labels['labels_to_ids_u']), num_proc=4)
            test_dataset = test_dataset.map(lambda e: convert_to_ids(e['u'], 'u', dcu_labels['labels_to_ids_u']), num_proc=4)

            ds = train_dataset['d']
            print(max(np.max(ds)))


        if cfg.model.config not in MODEL_CLASSES:
            raise ValueError('Please specify a valid model configuration.')

        logger.info('Loading encoder-decoder model.')
        encoder_class, decoder_class, tokenizer_class = MODEL_CLASSES[cfg.model.config]
        tokenizer = tokenizer_class.from_pretrained(cfg.model.tokenizer_name_or_path)
        encoder = encoder_class(vocab_size=cfg.model.vocab_size,
                                    pad_index=tokenizer.pad_token_id,
                                    **cfg.model.encoder_args)
        decoder = decoder_class(**cfg.model.decoder_args)
        model = TransformerEncoderDecoder(encoder=encoder,
                                          decoder=decoder,
                                          d_model=cfg.model.encoder_args.hidden_dim,
                                          vocab_size=cfg.model.vocab_size)
        if cfg.parallel:
            model = torch.nn.DataParallel(model)
        model.to(cfg.device)

    if cfg.run.do_train:
        if cfg.model.config == 'baseline_trans':
            train_baseline(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset
            )
        elif cfg.model.config == 'syntax_augmented_trans':
            train_syntax_augmented_trans(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset
            )

    if cfg.run.do_test:
        if cfg.model.config == 'baseline_trans':
            test_baseline(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset
            )
        elif cfg.model.config == 'syntax_augmented_trans':
            test_syntax_augmented_trans(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset
            )


if __name__ == '__main__':
    main()
