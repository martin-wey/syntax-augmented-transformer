import numpy as np
import torch

import networkx as nx

from data.code2ast import code2ast, get_tokens_ast, has_error
from data.binary_tree import ast2binary, tree_to_distance
from data.utils import get_c_subword, get_d_subword, get_u_subword


def filter_sample(code, lang, parser):
    try:
        G, code_pre = code2ast(code=code, parser=parser, lang=lang)
        assert nx.is_tree(nx.Graph(G))
        assert nx.is_connected(nx.Graph(G))
    except:
        return False
    if has_error(G):
        return False

    return True


def convert_sample_to_features(code, parser, lang):
    G, pre_code = code2ast(code, parser, lang)
    binary_ast = ast2binary(G)
    d, c, _, u = tree_to_distance(binary_ast, 0)
    code_tokens = get_tokens_ast(G, pre_code)

    return {
        'd': d,
        'c': c,
        'u': u,
        'num_tokens': len(code_tokens),
        'code': code_tokens
    }


def match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer):
    tokenized = []
    mapping = {}
    cont = 0
    for j, t in enumerate(untokenized_sent):
        if j == 0:
            temp = [k for k in tokenizer.tokenize(t) if k != 'Ġ']
            tokenized.append(temp)
            mapping[j] = [f for f in range(cont, len(temp) + cont)]
            cont = cont + len(temp)
        else:
            temp = [k for k in tokenizer.tokenize(' ' + t) if k != 'Ġ']
            tokenized.append(temp)
            mapping[j] = [f for f in range(cont, len(temp) + cont)]
            cont = cont + len(temp)
    flat_tokenized = [item for sublist in tokenized for item in sublist]
    return flat_tokenized, mapping


def collator_fn(batch, tokenizer, cfg):
    code_tokens_batch = [e['code'] for e in batch]
    docstrings_batch = [e['docstring'] for e in batch]

    all_input_ids = []
    for untokenized_sent in code_tokens_batch:
        to_convert, mapping = match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer)
        to_convert = to_convert[:cfg.model.max_input_length - 2]
        inputs = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + to_convert + [tokenizer.sep_token])
        all_input_ids.append(inputs)
    # padding
    all_input_ids = torch.tensor(
        [inputs + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * (cfg.model.max_input_length - len(inputs)))
         for inputs in all_input_ids])

    all_target_ids = tokenizer(
        docstrings_batch,
        padding='max_length',
        max_length=cfg.model.max_target_length,
        truncation=True,
        return_tensors='pt').input_ids

    src_padding_mask = (all_input_ids == tokenizer.pad_token_id)
    tgt_padding_mask = (all_target_ids == tokenizer.pad_token_id)

    return all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask


def collator_fn_dcu(batch, tokenizer, cfg):
    code_tokens_batch = [b['code'] for b in batch]
    docstrings_batch = [e['docstring'] for e in batch]
    cs = [b['c'] for b in batch]
    ds = [b['d'] for b in batch]
    us = [b['u'] for b in batch]

    # generate inputs and attention masks
    all_inputs = []
    all_mappings = []
    for untokenized_sent in code_tokens_batch:
        to_convert, mapping = match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer)
        to_convert = to_convert[:cfg.model.max_input_length - 2]
        inputs = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + to_convert + [tokenizer.sep_token])
        all_inputs.append(inputs)
        all_mappings.append({x: [l + 1 for l in y] for x, y in mapping.items()})
    # pad sequences
    all_input_ids = torch.tensor(
        [inputs + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * (cfg.model.max_input_length - len(inputs)))
         for inputs in all_inputs])

    all_target_ids = tokenizer(
        docstrings_batch,
        padding='max_length',
        max_length=cfg.model.max_target_length,
        truncation=True,
        return_tensors='pt').input_ids

    css, dss, uss = [], [], []
    for c, d, u, mapping in zip(cs, ds, us, all_mappings):
        css.append(get_c_subword(c, mapping)[:cfg.model.max_input_length - 1])
        dss.append(get_d_subword(d, mapping)[:cfg.model.max_input_length - 1])
        uss.append(get_u_subword(u, mapping)[:cfg.model.max_input_length])

    cs = torch.tensor([c + [255] * (cfg.model.max_input_length - 1 - len(c)) for c in css])
    ds = torch.tensor([d + [999] * (cfg.model.max_input_length - 1 - len(d)) for d in dss])
    us = torch.tensor([u + [255] * (cfg.model.max_input_length - len(u)) for u in uss])

    src_padding_mask = (all_input_ids == tokenizer.pad_token_id)
    tgt_padding_mask = (all_target_ids == tokenizer.pad_token_id)

    all_target_ids = tokenizer(
        docstrings_batch,
        padding='max_length',
        max_length=cfg.model.max_target_length,
        truncation=True,
        return_tensors='pt').input_ids

    return all_input_ids, src_padding_mask, all_target_ids, tgt_padding_mask, ds, cs, us
