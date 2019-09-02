# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
from logging import getLogger
import numpy as np
import torch

from ..utils import get_nn_avg_dist


DIC_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'crosslingual', 'dictionaries')


logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval, print_trans=False, result_path=''):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    if dico_eval == 'default':
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        path = dico_eval
    dico = load_dictionary(path, word2id1, word2id2)
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = torch.cat(word_scores, 1)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None] + average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    # retrieves the 10 top values along the first dimension, [1] accesses the index array, [0] is the array with the values
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 5, 10]:
        c = 0
        top_k_matches = top_matches[:, :k]
        logger.info(print_trans)
        if print_trans:
            logger.info(print_trans)
            matching_array =  (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches))
            # analyze the words that have no matching
            id2word2 = reverse_dict(word2id2)
            id2word1 = reverse_dict(word2id1)

            # load the full dictionary
            path_full_dict = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))

            dico_full = load_dictionary(path_full_dict, word2id1, word2id2)
            dico_full = dico_full.cuda() if emb1.is_cuda else dico_full
            dico_full_reversed = get_tgt2src(dico_full)
            transls = {}
            for i, w in enumerate(matching_array):
                if type(dico[i, 0]) == int:
                    tok = id2word1[dico[i, 0]]
                else:
                    tok = id2word1[dico[i, 0].item()]
                transls.setdefault(tok, {}).setdefault('matches', []).append(w.sum())

                # find the gold translation of the word
                if type(dico[i,1])  == int:
                    idx_gold = dico[i,1]
                else:
                    idx_gold = dico[i,1].item()
                gold = id2word2[idx_gold]
                transls.setdefault(tok, {}).setdefault('gold', set()).add(gold)
                # find the predicted translation of the word
                if type(top_k_matches[i,0]) == int:
                    idx_predicted = top_k_matches[i, :k]
                    predicted = [id2word2[idx] for idx in idx_predicted]
                else:
                    idx_predicted = top_k_matches[i, :k]
                    predicted = [id2word2[idx.item()] for idx in idx_predicted]

                transls.setdefault(tok, {}).setdefault('predictions', []).append(predicted)
                # find the english translations of those words

                gold_trans = set([id2word1[i] if i in id2word1.keys() and idx_gold in dico_full_reversed.keys() else 'NOT_FOUND' for i in dico_full_reversed[idx_gold] ])
                transls.setdefault(tok, {}).setdefault('gold_transls', set()).update(gold_trans)
                if idx_predicted in dico_full_reversed.keys():
                    predicted_trans = set([id2word1[i] for i in dico_full_reversed[idx_predicted]])
                else:
                    predicted_trans = set(['NO_TRANSL'])
                transls.setdefault(tok, {}).setdefault('predicted_transls', set()).update(predicted_trans)

            with open(os.path.join(result_path, 'translations_k{}.txt'.format(k)), 'w') as f:
                for tok, d in transls.items():
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(np.max(d['matches']),
                                                              tok,
                                                              ','.join(list(d['gold'])),
                                                              '#'.join(list(d['gold_transls'])),
                                                              ','.join(d['predictions']),
                                                              ','.join(list(d['predicted_transls']))))
            f.close()
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results


def reverse_dict(d):
    rev_d = {}
    for key, val in d.items():
        rev_d[val] = key
    return rev_d

def get_tgt2src(dico):
    '''
    mapping with values in second column as keys and values in first columns as values
    :param dico:
    :return:
    '''
    tgt2src = {}
    for i in dico:
        if type(i[1]) == int:
            tgt2src.setdefault(i[1], []).append(i[0])
        else:
            tgt2src.setdefault(i[1].item(), []).append(i[0].item())
    return tgt2src

