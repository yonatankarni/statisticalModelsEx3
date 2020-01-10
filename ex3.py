from collections import Counter
from time import time
import numpy as np


# calc_wti: calculate the probability for cluster i given doc t
# i: cluster index (0..8)
# clusters_probabilities - array, i-th place has the probability of a document belonging to cluster i (before we look
#   at the words)
# clusters_word_probabilities - array, i-th place has a map from words to their probabilities conditioned on cluster i
# yt - the th document (a counter over the document words)
def calc_wti(i, yt, clusters_probabilities, clusters_word_probabilities):
    n = len(clusters_probabilities)
    alpha_i = clusters_probabilities[i]
    document_piks = [pow(clusters_word_probabilities[word][i], cnt) for word, cnt in yt.most_common()]
    numerator = alpha_i * np.prod(document_piks)
    denom_list = np.zeros(n)
    for j in range(n):
        alpha_j = clusters_probabilities[j]
        cluster_document_piks = [pow(clusters_word_probabilities[word][j], cnt) for word, cnt in yt.most_common()]
        denom_list[j] = alpha_j * np.prod(cluster_document_piks)
    denominator = np.sum(denom_list)

    return numerator / denominator


# returns a list of size N (the number of documents), each list element represents the probability distribution over
# the clusters, so list of lists
def e_step(documents, clusters_probabilities, clusters_word_probabilities):
    return [[calc_wti(i, doc, clusters_probabilities, clusters_word_probabilities) for i in range(len(clusters_probabilities))]
            for doc in documents]


def calc_ai(cluster_idx, wts):
    n = float(len(wts))
    return (1 / n) * np.sum([wt[cluster_idx] for wt in wts])


# word_k - a word
# cluster_idx - the cluster # (0-8)
# documents - list of counters (with word frequencies for each document)
def calc_pik(word_k, cluster_idx, wts, documents):
    numerator = np.sum([wts[t][cluster_idx]*document[word_k] for t, document in enumerate(documents)])
    denominator = np.sum([wts[t][cluster_idx]*sum(document.values()) for t, document in enumerate(documents)])
    return numerator / denominator


def m_step(wts, documents, vocab):
    cluster_count = len(wts[0])
    clusters_probabilities = [calc_ai(cluster_idx, wts) for cluster_idx in range(cluster_count)]

    clusters_word_probabilities = {word_k: [calc_pik(word_k, cluster_idx, wts, documents) for cluster_idx in range(cluster_count)] for word_k in vocab}
    return clusters_probabilities, clusters_word_probabilities


def load_input(input_filename):
    with open(input_filename, 'r') as development_set_file:
        lines = development_set_file.readlines()

    vocab = Counter([word for line in lines[1::2] for word in line.strip().split(" ")]).keys()

    zipped = list(zip(lines[0::2], lines[1::2]))

    documents_info = list(map(lambda x: (x[0], x[1].strip().split(" "), Counter(x[1].strip().split(" "))), zipped))
    return documents_info, vocab


if __name__ == "__main__":
    documents_info, vocab = load_input("dataset/develop.txt")

    documents = list(map(lambda x: x[2], documents_info))

    # initialize - fake e step where each document gets a cluster by calculating the document index modulo 9
    EYE = np.eye(9)
    wts = [EYE[t % 9] for t in range(len(documents))]

    # for i in range(1):
    start_time = time()

    clusters_probabilities, clusters_word_probabilities = m_step(wts, documents, vocab)
    wts = e_step(documents, clusters_probabilities, clusters_word_probabilities)

    print("run took " + str((time() - start_time)) + " seconds")
