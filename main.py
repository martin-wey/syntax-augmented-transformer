import os
import random
import logging

import torch
import hydra
import omegaconf
import wandb
import numpy as np
from transformers import RobertaTokenizerFast
from datasets import load_dataset
from tree_sitter import Parser
from hydra.utils import get_original_cwd

from models.encoder_decoder import TransformerEncoder, TransformerDecoder, TransformerEncoderDecoder
from data.utils import LANGUAGE_GRAMMARS
from collator import filter_sample, convert_sample_to_features
from train import run_train_baseline

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'baseline_trans': (TransformerEncoder, TransformerDecoder, RobertaTokenizerFast),
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
    logger.info(f'Training on {torch.cuda.device_count()} GPUs.')
    logger.info(f'Run directory: {os.getcwd()}')

    wandb_cfg = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project='code-syntax-augmented-transformer', entity='mweyssow', config=wandb_cfg)
    cfg.run.output_path = os.path.join(cfg.run.base_path, cfg.run.name)
    if not os.path.exists(cfg.run.output_path):
        os.mkdir(cfg.run.output_path)

    # select the parser
    parser = Parser()
    parser.set_language(LANGUAGE_GRAMMARS[cfg.run.lang])

    if cfg.run.do_train or cfg.run.do_test:
        if cfg.model.config not in MODEL_CLASSES:
            raise ValueError('Please specify a valid model configuration.')

        if cfg.model.config == 'baseline_trans':
            logger.info('Loading encoder-decoder baseline.')
            encoder_class, decoder_class, tokenizer_class = MODEL_CLASSES[cfg.model.config]
            tokenizer = tokenizer_class.from_pretrained(cfg.model.tokenizer_name_or_path)
            encoder = encoder_class(vocab_size=cfg.model.vocab_size,
                                    hidden_dim=cfg.model.encoder_hidden_dim,
                                    num_heads=cfg.model.encoder_num_heads,
                                    num_layers=cfg.model.encoder_num_layers,
                                    dim_feedforward=cfg.model.encoder_ff_hidden_dim,
                                    dropout=cfg.model.encoder_dropout,
                                    pad_index=tokenizer.pad_token_id)
            decoder = decoder_class(hidden_dim=cfg.model.decoder_hidden_dim,
                                    num_heads=cfg.model.decoder_num_heads,
                                    num_layers=cfg.model.decoder_num_layers,
                                    dim_feedforward=cfg.model.decoder_ff_hidden_dim,
                                    dropout=cfg.model.decoder_dropout)
            model = TransformerEncoderDecoder(encoder=encoder,
                                              decoder=decoder,
                                              d_model=cfg.model.encoder_hidden_dim,
                                              vocab_size=cfg.model.vocab_size)
            if cfg.parallel:
                model = torch.nn.DataParallel(model)
            model.to(cfg.device)

        logger.info('Loading train/valid datasets.')
        dataset_path = os.path.join(cfg.run.dataset_dir, cfg.run.lang)
        data_files = {
            'train': os.path.join(get_original_cwd(), dataset_path, 'train.jsonl'),
            'valid': os.path.join(get_original_cwd(), dataset_path, 'valid.jsonl')
        }
        train_dataset = load_dataset('json', data_files=data_files, split='train')
        valid_dataset = load_dataset('json', data_files=data_files, split='valid')

        # filter codes that cannot be parsed with tree_sitter
        logger.info('Filtering datasets.')
        train_dataset = train_dataset.filter(lambda e: filter_sample(e['original_string'], cfg.run.lang, parser))
        valid_dataset = valid_dataset.filter(lambda e: filter_sample(e['original_string'], cfg.run.lang, parser))

        train_dataset = train_dataset.map(lambda e: convert_sample_to_features(e['original_string'], parser, cfg.run.lang))
        valid_dataset = valid_dataset.map(lambda e: convert_sample_to_features(e['original_string'], parser, cfg.run.lang))

    if cfg.run.do_train:
        if cfg.model.config == 'baseline_trans':
            run_train_baseline(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset
            )


if __name__ == '__main__':
    main()
