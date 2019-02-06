import sys
from pyswarm import pso
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
import argparse
from extensions.my_utils import normalize_embeddings
from extensions.inspect_embeddings import load_embeddings_from_file
from sklearn.decomposition import PCA
import logging
import os
import numpy as np



class SWUM(object):

    def __init__(self, params, src_emb, trg_emb, init_method):
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.dim = src_emb.shape[1]
        self.params = params
        self.init_method = init_method
        self.lb = self.generate_init_mapping(self.dim, -params.space, method=self.init_method)
        self.ub = self.generate_init_mapping(self.dim, params.space, method=self.init_method)


    def nn(self, Transformed, Fr):
        T=cKDTree(Fr)
        Dists,NeighborIDs=T.query(Transformed)
        return NeighborIDs


    def loss(self, Tr_onedim): # Tr a linear transformation
        """
        compute the loss for the given parameters as the sum over cosine distances between nearest neighbors
        :param Tr_onedim:
        :return:
        """
        # make a 300 x 300 matrix from the 90000 dimensional Tr_onedim
        Tr = np.reshape(Tr_onedim, (self.dim,self.dim))
        transformed_src = self.transform(Tr, self.src_emb)
        mappedIDs=self.nn(transformed_src ,self.trg_emb)
        mappedNNs=self.trg_emb[mappedIDs]

        dists = cosine_distances(transformed_src, mappedNNs)
        # average the diagonal values of this array
        lossval = np.mean(np.diag(dists))
        logging.info(lossval)
        return lossval


    def transform(self, mapping, values):
        """
        transform the values by applying the linear mapping
        :param mapping:
        :param values:
        :return:
        """
        return np.matmul(values, mapping)

    def generate_init_mapping(self, dim, init_val, method):
        """
        the mapping is represented as a one-dimensional array of length dim**2
        :param dim:
        :param init_val:
        :return:
        """
        if method == 'artexte2018':
            mapping = compute_initial_mapping_artetxe(self.src_emb, self.trg_emb).flatten() * init_val
        else:
            mapping = [1 * init_val for i in range(dim**2)]
        return mapping


    def run(self, debug=False):
        xopt, fopt = pso(self.loss, self.lb, self.ub, swarmsize=self.params.size, phig=self.params.phig, debug=debug)
        return xopt

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
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

def generate_synthetic_toy_data(num_words1, num_words2, dim):
    emb1 = np.reshape(np.random.rand(num_words1*dim), (num_words1, dim))
    #emb2 = np.reshape(np.random.rand((num_words2-num_words1)*dim), ((num_words2-num_words1), dim))
    #emb2 = np.vstack([emb1, emb2])
    emb2=emb1
    mapping = np.reshape(np.random.rand(dim*dim), (dim, dim))
    return emb1, np.matmul(emb2, mapping)

def generate_synthetic_data(emb_file, dim,max_vocab):
    # read embeddings, generate a linear mapping and return mapped embeddings. to test if algorithm can recover the mapping
    # given perfect isometry
    dico, embs = load_embeddings_from_file(emb_file, 'en', max_vocab)
    mapping = np.reshape(np.random.rand(dim * dim), (dim, dim))
    return embs, np.matmul(embs, mapping)

def perform_pca(n_components, embs):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(embs)
    return transformed

def compute_initial_mapping_artetxe(x, z, unsupervised_vocab=4000, csls_neighborhood=10):
    """
    compute a mapping according to Artetxe (2018)
    :return:
    """
    sim_size = min(x.shape[0], z.shape[0]) if unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], unsupervised_vocab)
    u, s, vt = np.linalg.svd(x[:sim_size], full_matrices=False)
    xsim = (u * s).dot(u.T)
    u, s, vt = np.linalg.svd(z[:sim_size], full_matrices=False)
    zsim = (u * s).dot(u.T)
    del u, s, vt
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    #normalize(xsim, normalize)
    #normalize(zsim, normalize)

    sim = xsim.dot(zsim.T)
    knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
    knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
    sim -= knn_sim_fwd[:, np.newaxis] / 2 + knn_sim_bwd / 2
    src_indices = np.concatenate((np.arange(sim_size), sim.argmax(axis=0)))
    trg_indices = np.concatenate((sim.argmax(axis=1), np.arange(sim_size)))
    mapping = procrustes(z[trg_indices,:], x[src_indices,:])
    return mapping

def procrustes(emb1, emb2):
    u, s, vt = np.linalg.svd(emb1.T.dot(emb2))
    w = vt.T.dot(u.T)
    return w

def set_up_logging(exp_path):
    ######################################################################################################
    ################################# LOGGING ############################################################
    ######################################################################################################
    # create a logger and set parameters
    logfile = os.path.join(exp_path, 'log.txt')
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

def main(params):
    set_up_logging('tmp')

    # _,src_emb=load_embeddings_from_file(params.src_emb, lang='',max_vocab=params.max_vocab )
    # _,trg_emb=load_embeddings_from_file(params.tgt_emb,lang='',max_vocab=params.max_vocab)
    #src_emb, trg_emb = generate_synthetic_toy_data(10000, 10000, 30)
    emb_file = '/home/mareike/PycharmProjects/breakit/embeddings/embs/fastText/10_1_300.vec'
    src_emb, trg_emb = generate_synthetic_data(emb_file, params.emb_dim, max_vocab=10000)

    #src_emb = normalize_embeddings(src_emb, params.normalize_embeddings)
    #trg_emb = normalize_embeddings(trg_emb, params.normalize_embeddings)

    logging.info('perform pca on src embeddings')
    src_emb = perform_pca(50, src_emb)
    logging.info('perform pca on trg embeddings')
    trg_emb = perform_pca(50, trg_emb)

    swum = SWUM(params, src_emb, trg_emb, params.init_method)

    xopt = swum.run(debug=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--src_emb", type=str,  help="Load source embeddings")
    parser.add_argument("--tgt_emb", type=str,  help="Load target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximal size of the vocabulary")


    parser.add_argument("--size", type=int, default=100, help="Swarm size")

    parser.add_argument("--phig", type=float, default=0.9, help="Scaling factor to search away from the swarm’s best known position")
    parser.add_argument("--space", type=int, default=1, help="Search space scaling")
    parser.add_argument("--emb_dim", type=int, default=300, help="Search space scaling")
    parser.add_argument("--method", type=str, default='cosine', choices=['cosine', 'eigen'], help="Similarity to be optimized")
    parser.add_argument("--init_method", type=str, default='uniform', choices=['artetxe2018', 'uniform'],
                        help="Method to initialize the mapping")

    # parse parameters
    params = parser.parse_args()
    main(params)








