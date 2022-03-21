import logging
from collections import Counter

import torch
from torch.utils.data import RandomSampler

logger = logging.getLogger(__name__)


def get_random_sampler(dataset, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler


def get_grouped_params(model, weight_decay, lr, no_decay=None, no_grad_layers=[]):
    if no_decay is None:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if hasattr(model, 'module'):
        model = model.module

    return [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and n not in no_grad_layers],
            'weight_decay': weight_decay, 'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and n not in no_grad_layers],
            'weight_decay': 0.0, 'lr': lr,
        }
    ]


def num_trainable_parameters(model, debug=False):
    model = model.module if hasattr(model, 'module') else model
    if debug:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
            else:
                logger.info('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    return num_params


def get_non_terminals_labels(label_sets):
    all_labels = []
    for labels in label_sets:
        all_labels += [label for seq in labels for label in seq]
    # use a Counter to constantly get the same order in the labels
    ct = Counter(all_labels)
    labels_to_ids = {}
    for i, label in enumerate(ct):
        labels_to_ids[label] = i
    return labels_to_ids


def convert_to_ids(c, column_name, labels_to_ids):
    labels_ids = []
    for label in c:
        labels_ids.append(labels_to_ids[label])
    return {column_name: labels_ids}
