'''
provides methods to load a mapping between embeddings an compute the loss of the discriminator for that given mapping
'''

import os
import argparse
from collections import OrderedDict
import numpy as np

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

import torch
# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2




def oned_linear_interpolation(m_init, m_final, alpha):
    '''
    computes linear interpolation of the parameters
    :return:
    '''
    # get the weights of the paramter tensors as numpy arrays
    p_init = m_init.numpy()
    p_final = m_final.numpy()
    p_interpol = (1-alpha)*p_init + alpha * p_final
    # generate tensor
    return torch.from_numpy(p_interpol)


def sample_from_multivariate_gaussian(mean, var=1):
    """
    sample independent parameters in a parameter matrix from a multivariate gaussian centered on mean
    :param mean: the mean of the gaussian
    :param var: the variance of the gaussian. default 1
    :return:
    """
    mean = mean.numpy()
    sampled = []
    for row in mean:
        sampled.append(np.random.multivariate_normal(row.ravel(), np.identity(row.ravel().shape[0]) * var).reshape((row.shape)))
    return torch.from_numpy(np.array(sampled))


# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings

parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--mapping_i", type=str, default="", help="Path to the initial mapping to be loaded")
parser.add_argument("--mapping_f", type=str, default="", help="Path to the final mapping to be loaded")



# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
parser.add_argument("--noise", type=float, default=0, help="Variance of the noise to be added to the inputs (0 to disable)")



parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
parser.add_argument("--discriminator", type=str, default="", help="Path to the discriminator parameters to be loaded")

parser.add_argument("--interpolation_step_size", type=float, default=0.1, help="The stepsize for sampling mappings along the path from initial to target mapping")
parser.add_argument("--num_samples", type=int, default=0, help="The number of samples of mapping parameters drawn at each interpolation step. If 0 only the line segment is considered.")
parser.add_argument("--var", type=float, default=1., help="The variance of the gaussian distribution from which the samples are drawn.")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
discriminator.load(params.discriminator)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)

# Iteratively modify the mapping from initial mapping to target mapping by linear interpolation
# track the discriminator loss for all different mappings that ly between initial and target mapping
mapping_i = torch.from_numpy(torch.load(params.mapping_i))
mapping_f = torch.from_numpy(torch.load(params.mapping_f))
for alpha in np.arange(0, 1+params.interpolation_step_size, params.interpolation_step_size):
    interpolated_mapping = oned_linear_interpolation(mapping_i, mapping_f, alpha)

    mappings = [interpolated_mapping]

    for n in range(params.num_samples):
        mappings.append(sample_from_multivariate_gaussian(interpolated_mapping, var=params.var))

    for m, sampled_mapping in enumerate(mappings):
        trainer.set_mapping_weights(weights=sampled_mapping)

        # compute the discriminator loss
        logger.info('----> COMPUTING DISCRIMINATOR LOSS <----\n\n')
        logger.info('alpha={}, m={}'.format(alpha,m))
        losses = []
        for n_iter in range(0, params.epoch_size, params.batch_size):
            loss = trainer.compute_loss()
            losses.append(loss.cpu().data.numpy())
            #print(losses)
        logger.info('Discriminator loss {}: {}\n'.format(m, np.mean(losses)))

        evaluator = Evaluator(trainer)

        # run evaluations
        to_log = OrderedDict({'n_iter': 0})
        evaluator.monolingual_wordsim(to_log)
        # evaluator.monolingual_wordanalogy(to_log)
        if params.tgt_lang:
            evaluator.crosslingual_wordsim(to_log)
            evaluator.word_translation(to_log)
            #evaluator.sent_translation(to_log)
            evaluator.dist_mean_cosine(to_log)
