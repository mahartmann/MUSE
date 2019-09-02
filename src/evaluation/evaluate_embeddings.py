import os
import io
from scipy.stats import spearmanr
import numpy as np

def get_wordsim_scores(language, word2id, embeddings, lower=True):
    """
    Return monolingual word similarity scores.
    """
    dirpath = os.path.join(MONOLINGUAL_EVAL_PATH, language)
    if not os.path.isdir(dirpath):
        return None

    scores = {}
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logger.info(separator)
    logger.info(pattern % ("Dataset", "Found", "Not found", "Rho"))
    logger.info(separator)

    for filename in list(os.listdir(dirpath)):
        if filename.startswith('%s_' % (language.upper())):
            filepath = os.path.join(dirpath, filename)
            coeff, found, not_found = get_spearman_rho(word2id, embeddings, filepath, lower)
            logger.info(pattern % (filename[:-4], str(found), str(not_found), "%.4f" % coeff))
            scores[filename[:-4]] = coeff
    logger.info(separator)

    return scores

def get_word_id(word1, word2id1, lower):
    if lower is True:
        if word1.lower() not in word2id1:
            return None
        else:
            return word2id1[word1.lower()]
    else:
        if word1 not in word2id1:
            return None
        else:
            return word2id1[word1]


def get_spearman_rho_mono(word2id1, embeddings1, path, lower):
    """
    Compute monolingual or cross-lingual word similarity score.
    """
    word2id2 = word2id1
    embeddings2 = embeddings1
    assert len(word2id1) == embeddings1.shape[0]
    assert len(word2id2) == embeddings2.shape[0]
    assert type(lower) is bool
    word_pairs = get_word_pairs(path)
    not_found = 0
    pred = []
    gold = []
    for word1, word2, similarity in word_pairs:
        id1 = get_word_id(word1, word2id1, lower)
        id2 = get_word_id(word2, word2id2, lower)
        if id1 is None or id2 is None:
            not_found += 1
            continue
        u = embeddings1[id1]
        v = embeddings2[id2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)

    return gold, pred, len(gold), not_found #spearmanr(gold, pred).correlation, len(gold), not_found



def get_word_pairs(path, lower=True):
    """
    Return a list of (word1, word2, score) tuples from a word similarity file.
    """
    assert os.path.isfile(path) and type(lower) is bool
    word_pairs = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            line = line.lower() if lower else line
            line = line.split()
            # ignore phrases, only consider words
            if len(line) != 3:
                assert len(line) > 3
                assert 'SEMEVAL17' in os.path.basename(path) or 'EN-IT_MWS353' in path
                continue
            word_pairs.append((line[0], line[1], float(line[2])))
    return word_pairs

def load_embeddings_from_file(fname, max_vocab=-1):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    with open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue
            else:
                word, vect = line.rstrip().split(' ', 1)

                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                assert word not in word2id
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if max_vocab > 0 and i >= max_vocab:
                break

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    return embeddings, word2id, id2word

if __name__=="__main__":
    path = '/home/mareike/PycharmProjects/breakit/MUSE/data/monolingual/it/IT_SIMLEX-999.txt'
    emb_file = '/home/mareike/PycharmProjects/wiki2vec/code/code/wiki2svd_w2_neg5_300_it.svd.txt'
    word_pairs = get_word_pairs(path)
    embeddings, word2id, id2word = load_embeddings_from_file(emb_file)
    print(word_pairs)

    get_spearman_rho_mono(word2id, embeddings, path, lower=True)
