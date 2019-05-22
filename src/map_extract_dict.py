# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Maps and extracts a dict from the mapped embeddings

import os
import argparse
from collections import OrderedDict

import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

# main
parser = argparse.ArgumentParser(description='Extraction of dict')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")

parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=4000, help="Maximum generated dictionary size (0 to disable)")

# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload orig source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload orig target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="center", help="Normalize embeddings before training")

parser.add_argument("--mapping", type=str, default="", help="Path to the mapping to be loaded")
parser.add_argument("--outfile", type=str, default="", help="output file for dico")
parser.add_argument("--num_seeds", type=int, default=4000, help="Size of extracted dico")




# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)

loaded_mapping = torch.from_numpy(torch.load(params.mapping))
trainer.set_mapping_weights(weights=loaded_mapping)

# extract new dico
trainer.build_dictionary()
d = trainer.dico
d = d.numpy()
print(d)

with open(params.outfile, 'w') as f:
    for src, trg in d[:params.num_seeds, :]:
        print(trainer.src_dico.id2word[src], trainer.tgt_dico.id2word[trg])
        f.write('{} {}\n'.format(trainer.src_dico.id2word[src], trainer.tgt_dico.id2word[trg]))
f.close()

evaluator = Evaluator(trainer)

# run evaluations
to_log = OrderedDict({'n_iter': 0})
evaluator.monolingual_wordsim(to_log)
#evaluator.monolingual_wordanalogy(to_log)
if params.tgt_lang:
    evaluator.crosslingual_wordsim(to_log)
    evaluator.word_translation(to_log)
    evaluator.sent_translation(to_log)
    #evaluator.dist_mean_cosine(to_log)

