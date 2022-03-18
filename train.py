import logging
import os

import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from utils import get_random_sampler, num_trainable_parameters
from collator import collator_fn

logger = logging.getLogger(__name__)


def run_train_baseline(
    cfg,
    model,
    tokenizer,
    train_dataset,
    valid_dataset
):
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
            all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask = batch

            logits = model(all_input_ids.to(cfg.device),
                           all_target_ids.to(cfg.device),
                           src_padding_mask.to(cfg.device),
                           tgt_padding_mask.to(cfg.device),
                           src_padding_mask.to(cfg.device))

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
                wandb.log({'train/loss': avg_loss, 'step': step * epoch, 'epoch': epoch})
                logger.info(f'\nepoch {epoch} | step {step} | loss {avg_loss}')
                step_loss, step_num = 0, 0

        training_loss /= len(train_dataloader)
        eval_loss = evaluate_baseline(valid_dataloader, model, criterion, cfg)
        wandb.log({'validation/loss': round(eval_loss, 4), 'epoch': epoch})
        logger.info(f'epoch #{epoch} | training loss {round(training_loss, 4)} | validation loss {round(eval_loss, 4)}')
        model.train()

        if eval_loss < best_eval_loss:
            logger.info('-' * 100)
            logger.info('Saving model checkpoint')
            logger.info('-' * 100)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_path = os.path.join(cfg.run.output_path, 'pytorch_model.bin')
            torch.save(model_to_save.state_dict(), output_path)
            logger.info(f'New model saved: {output_path}')
            patience_count = 0
            best_eval_loss = eval_loss
            wandb.run.summary['best_validation_loss'] = best_eval_loss
        else:
            patience_count += 1

        if patience_count == cfg.run.patience:
            logger.info('-' * 100)
            logger.info(f'Stopping training (out of patience, patience={cfg.run.patience})')
            logger.info('-' * 100)
            break


def evaluate_baseline(
    valid_dataloader,
    model,
    criterion,
    cfg
):
    logger.info('***** Running validation *****')
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader,
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}',
                                          desc='Iteration')):
            all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask = batch

            logits = model(all_input_ids.to(cfg.device),
                           all_target_ids.to(cfg.device),
                           src_padding_mask.to(cfg.device),
                           tgt_padding_mask.to(cfg.device),
                           src_padding_mask.to(cfg.device))

            tgt_shifted = all_target_ids[:, 1:]
            logits_shifted = logits[:, :-1, :]
            loss = criterion(logits_shifted.reshape(-1, logits_shifted.shape[-1]).to(cfg.device),
                             tgt_shifted.reshape(-1).to(cfg.device))
            eval_loss += loss.item()
    return eval_loss / len(valid_dataloader)
