"""
methods to compute an initial mapping that can be used to initialize the generator instead of identity
"""
import numpy as np
from src.utils import load_embeddings, normalize_embeddings
import argparse
import torch

def extract_initial_mapping(params, src_embs, trg_embs):
    # compute 2nd order matrices for each embeddings
    # sort the values in each row
    # do neareset neighbor search to determine initial matches
    # modify the identity matrix such that these points will be mapped onto each other
    # mapping can be learned by solving Procrustes

    src_indices, trg_indices = build_seed_dictionary(params, src_embs, trg_embs)
    #src_indices = (src_indices.cuda() if params.cuda else src_indices)
    #trg_indices = (trg_indices.cuda() if params.cuda else trg_indices)

    src_embs_aligned = src_embs.weight.data[src_indices, :]
    trg_embs_aligned = trg_embs.weight.data[trg_indices, :]
    mapping_init = procrustes(src_embs_aligned, trg_embs_aligned)
    return mapping_init


def build_seed_dictionary(params, src_embs_torch, trg_embs_torch, csls_neighborhood=1, direction='forward', unsupervised_vocab=0):
    """
    This code is taken from artetxe's vecmap repository
    :return:
    """
    # normalize embeddings
    normalize_embeddings(src_embs_torch.weight.data, params.normalize_embeddings)
    normalize_embeddings(trg_embs_torch.weight.data, params.normalize_embeddings)

    src_emb = src_embs_torch.weight.data
    trg_emb = trg_embs_torch.weight.data
    # Build the seed dictionary
    src_indices = []
    trg_indices = []

    #sim_size = min(src_emb.shape[0], trg_emb.shape[0]) if unsupervised_vocab <= 0 else min(src_emb.shape[0], trg_emb.shape[0], unsupervised_vocab)
    sim_size = min(20000, 20000) if unsupervised_vocab <= 0 else min(src_emb.shape[0],
                                                                                           trg_emb.shape[0],
                                                                                           unsupervised_vocab)
    u, s, vt = np.linalg.svd(src_emb[:sim_size], full_matrices=False)
    xsim = (u * s).dot(u.T)

    u, s, vt = np.linalg.svd(trg_emb[:sim_size], full_matrices=False)
    zsim = (u * s).dot(u.T)
    del u, s, vt
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    sim = xsim.dot(zsim.T)

    if csls_neighborhood > 0:
        knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
        knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
        sim -= knn_sim_fwd[:, np.newaxis] / 2 + knn_sim_bwd / 2
    if direction == 'forward':
        src_indices = np.arange(sim_size)
        trg_indices = sim.argmax(axis=1)
    elif direction == 'backward':
        src_indices = sim.argmax(axis=0)
        trg_indices = np.arange(sim_size)
    elif direction == 'union':
        src_indices = np.concatenate((np.arange(sim_size), sim.argmax(axis=0)))
        trg_indices = np.concatenate((sim.argmax(axis=1), np.arange(sim_size)))
    del xsim, zsim, sim

    return src_indices, trg_indices


def topk_mean(m, k, inplace=False):
    """
    taken from vecmap
    """
    # TODO Assuming that axis is 1
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def procrustes(src_embs_aligned, trg_embs_aligned):
    """
    Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """

    u, s, vt = np.linalg.svd(np.matmul(np.transpose(src_embs_aligned), trg_embs_aligned), full_matrices=False)
    return u.dot(vt)






if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--src_emb", type=str, default='/home/mareike/PycharmProjects/breakit/embeddings/test_emb.vec')
    parser.add_argument("--tgt_emb", type=str,
                        default='/home/mareike/PycharmProjects/breakit/embeddings/test_emb.vec')
    parser.add_argument("--src_lang", type=str,
                        default='en')
    parser.add_argument("--tgt_lang", type=str,
                        default='en')
    parser.add_argument("--emb_dim", type=int,
                        default=300)
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--normalize_embeddings", type=str, default="center")

    # parse parameters
    params = parser.parse_args()

    src_dico, src_emb = load_embeddings(params, source=True)
    trg_dico, trg_emb = load_embeddings(params, source=False)
    #src_emb = src_emb.numpy()
    #trg_emb = trg_emb.numpy()





    m = extract_initial_mapping(src_emb.numpy(), trg_emb.numpy())


