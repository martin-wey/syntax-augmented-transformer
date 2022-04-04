import os
import logging
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from collators import collator_fn, collator_fn_dcu
from models.encoder_decoder import create_mask, Beam
import models.bleu as bleu

logger = logging.getLogger(__name__)


def save_predictions(p, test_dataset, cfg):
    save_path = cfg.model.checkpoint if cfg.model.checkpoint is not None else None
    logger.info('Saving predictions.')
    predictions = []
    golds = []
    with open(os.path.join(save_path, 'test.pkl'), 'wb') as f1, \
            open(os.path.join(save_path, 'predictions.pkl'), 'wb') as f2:
        for ref, (idx, gold) in zip(p, enumerate(test_dataset)):
            predictions.append(ref)
            golds.append(gold["docstring"])
        pickle.dump(golds, f1)
        pickle.dump(predictions, f2)


def compute_beam_search_batch(encoder_output, all_input_ids, src_mask, tokenizer, model):
    preds = []
    zero = torch.cuda.LongTensor(1).fill_(0)
    for i in range(all_input_ids.shape[0]):
        # shape (seq_len, 1, hidden_dim)
        context = encoder_output[:, i:i + 1]
        # shape (1, seq_len)
        context_mask = src_mask[i:i + 1, :]
        beam = Beam(size=10, sos=tokenizer.bos_token_id, eos=tokenizer.eos_token_id)
        input_ids = beam.getCurrentState()
        # shape (seq_len, beam_size, hidden_dim)
        context = context.repeat(1, 10, 1)
        # shape (beam_size, seq_len)
        context_mask = context_mask.repeat(10, 1)
        for _ in range(50):
            if beam.done(): break
            out = model.predict(input_ids, context, context_mask)
            beam.advance(out)
            input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
        hyp = beam.getHyp(beam.getFinal())
        pred = beam.buildTargetTokens(hyp)[:10]
        pred = [torch.cat([x.view(-1) for x in p] + [zero] * (50 - len(p))).view(1, -1) for p in pred]
        preds.append(torch.cat(pred, 0).unsqueeze(0))

    preds = torch.cat(preds, 0)
    batch_list = []
    for pred in preds:
        t = pred[0].cpu().numpy()
        t = list(t)
        if 0 in t:
            t = t[:t.index(0)]
        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
        batch_list.append(text)
    return batch_list


def test_baseline(cfg, model, tokenizer, test_dataset):
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=cfg.run.valid_batch_size,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer, cfg),
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=8)
    model.eval()
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}',
                                          desc='Iteration')):
            all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask = batch
            src_mask, _ = create_mask(all_input_ids.to(cfg.device), None)
            encoder_output = model.encoder(all_input_ids.to(cfg.device), src_mask, src_padding_mask.to(cfg.device))
            batch_list = compute_beam_search_batch(encoder_output, all_input_ids, src_mask, tokenizer, model)
            predictions += batch_list
    save_predictions(predictions, test_dataset, cfg)


def test_syntax_augmented_trans(cfg, model, tokenizer, test_dataset):
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=cfg.run.valid_batch_size,
                                 collate_fn=lambda batch: collator_fn_dcu(batch, tokenizer, cfg),
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=8)
    model.eval()
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}',
                                          desc='Iteration')):
            all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask, ds, cs, us = batch
            src_mask, _ = create_mask(all_input_ids.to(cfg.device), None)
            encoder_output = model.encoder(all_input_ids.to(cfg.device), src_mask, src_padding_mask.to(cfg.device),
                                           ds.to(cfg.device), cs.to(cfg.device), us.to(cfg.device))
            batch_list = compute_beam_search_batch(encoder_output, all_input_ids, src_mask, tokenizer, model)
            predictions += batch_list
    save_predictions(predictions, test_dataset, cfg)


def compute_bleu(predictions_file, gold_file):
    with open(predictions_file, 'rb') as f1, open(gold_file, 'rb') as f2:
        predictions = pickle.load(f1)
        golds = pickle.load(f2)

    preds, refs = [], []
    for pred, gold in zip(predictions, golds):
        preds.append(bleu.splitPuncts(pred.strip().lower()))
        refs.append(bleu.splitPuncts(pred.strip().lower()))

    score = [0] * 5
    num = 0.0
    for pred, ref in zip(preds, refs):
        bl = bleu.bleu(pred, ref)
        score = [score[i] + bl[i] for i in range(0, len(bl))]
        num += 1
    bleu_score = [s * 100.0 / num for s in score][0]
    logger.info(f'  bleu-4 = {bleu_score}')
    logger.info('  ' + '*' * 20)
