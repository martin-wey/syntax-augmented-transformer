import logging

import torch
import wandb
import omegaconf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizerFast
from datasets import Dataset
from tqdm import tqdm

from models.encoder_decoder import TransformerEncoderDecoder
from utils import get_random_sampler, num_trainable_parameters
from collators import collator_fn_seq2seq, collator_fn_dcu_seq2seq

logger = logging.getLogger(__name__)


def train_seq2seq(
    cfg: omegaconf.DictConfig,
    model: TransformerEncoderDecoder,
    tokenizer: RobertaTokenizerFast,
    train_dataset: Dataset,
    valid_dataset: Dataset
):
    collator_fn = collator_fn_seq2seq if cfg.model.type == 'baseline' else collator_fn_dcu_seq2seq
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg.run.train_batch_size,
                                  sampler=get_random_sampler(train_dataset, cfg.run.seed),
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer, cfg),
                                  pin_memory=True,
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=cfg.run.valid_batch_size,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer, cfg),
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=8)

    optimizer = AdamW(model.module.parameters() if hasattr(model, 'module') else model.parameters(),
                      lr=cfg.run.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    logger.info('***** Running training *****')
    logger.info(f'  Num trainable parameters = {num_trainable_parameters(model)}')
    logger.info(f'  Num train examples = {len(train_dataset)}')
    logger.info(f'  Num valid examples = {len(valid_dataset)}')
    logger.info(f'  Num Epochs = {cfg.run.epochs}')
    logger.info(f'  Total train batch size = {cfg.run.train_batch_size}')
    logger.info(f'  Total optimization steps = {len(train_dataloader) * cfg.run.epochs}')

    model.train()
    best_eval_loss = float('inf')
    patience_count = 0
    for epoch in range(1, cfg.run.epochs + 1):
        training_loss = 0
        step_loss, step_num = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader,
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}',
                                          desc='Iteration')):
            if cfg.model.type == 'baseline':
                all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask = batch
                logits = model(all_input_ids.to(cfg.device),
                               all_target_ids.to(cfg.device),
                               src_padding_mask.to(cfg.device),
                               tgt_padding_mask.to(cfg.device),
                               src_padding_mask.to(cfg.device))
            else:
                all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask, ds, cs, us = batch
                logits = model(all_input_ids.to(cfg.device),
                               all_target_ids.to(cfg.device),
                               src_padding_mask.to(cfg.device),
                               tgt_padding_mask.to(cfg.device),
                               src_padding_mask.to(cfg.device),
                               ds.to(cfg.device),
                               cs.to(cfg.device),
                               us.to(cfg.device))

            tgt_shifted = all_target_ids[:, 1:]
            logits_shifted = logits[:, :-1, :]
            loss = criterion(logits_shifted.reshape(-1, logits_shifted.shape[-1]).to(cfg.device),
                             tgt_shifted.reshape(-1).to(cfg.device))

            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # report loss
            training_loss += loss.item()
            step_loss += loss.item()
            step_num += 1
            if step % 100 == 0 and step > 0:
                avg_loss = round(step_loss / step_num, 4)
                logger.info(f'\nepoch {epoch} | step {step} | loss {avg_loss}')
                if cfg.use_wandb:
                    wandb.log({'train/loss': avg_loss, 'step': step * epoch, 'epoch': epoch})
                step_loss, step_num = 0, 0

        training_loss /= len(train_dataloader)
        eval_loss = evaluate_seq2seq(cfg, valid_dataloader, model, criterion)
        logger.info(f'epoch #{epoch} | training loss {round(training_loss, 4)} | validation loss {round(eval_loss, 4)}')
        if cfg.use_wandb:
            wandb.log({'validation/loss': round(eval_loss, 4), 'epoch': epoch})
        model.train()

        if eval_loss < best_eval_loss:
            logger.info('-' * 100)
            logger.info('Saving model checkpoint')
            logger.info('-' * 100)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'pytorch_model.bin')
            logger.info(f'New best model saved!')
            patience_count = 0
            best_eval_loss = eval_loss
            if cfg.use_wandb:
                wandb.run.summary['best_validation_loss'] = best_eval_loss
        else:
            patience_count += 1

        if patience_count == cfg.run.patience:
            logger.info('-' * 100)
            logger.info(f'Stopping training (out of patience, patience={cfg.run.patience})')
            logger.info('-' * 100)
            break


def evaluate_seq2seq(
    cfg: omegaconf.DictConfig,
    valid_dataloader: DataLoader,
    model: TransformerEncoderDecoder,
    criterion: torch.nn.CrossEntropyLoss,
):
    logger.info('***** Running validation *****')
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader,
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}',
                                          desc='Iteration')):
            if cfg.model.type == 'baseline':
                all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask = batch
                logits = model(all_input_ids.to(cfg.device),
                               all_target_ids.to(cfg.device),
                               src_padding_mask.to(cfg.device),
                               tgt_padding_mask.to(cfg.device),
                               src_padding_mask.to(cfg.device))
            else:
                all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask, ds, cs, us = batch
                logits = model(all_input_ids.to(cfg.device),
                               all_target_ids.to(cfg.device),
                               src_padding_mask.to(cfg.device),
                               tgt_padding_mask.to(cfg.device),
                               src_padding_mask.to(cfg.device),
                               ds.to(cfg.device),
                               cs.to(cfg.device),
                               us.to(cfg.device))

            tgt_shifted = all_target_ids[:, 1:]
            logits_shifted = logits[:, :-1, :]
            loss = criterion(logits_shifted.reshape(-1, logits_shifted.shape[-1]).to(cfg.device),
                             tgt_shifted.reshape(-1).to(cfg.device))
            eval_loss += loss.item()
    return eval_loss / len(valid_dataloader)


def train_encoder():
    pass


def evaluate_encoder():
    pass
