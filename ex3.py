from collections import Counter
from time import time
import numpy as np


underflow_k = 10
epsilon = 1e-10
pik_lambda = 1e-10


# calc_wti: calculate the probability for cluster i given doc t
# i: cluster index (0..8)
# clusters_probabilities - array, i-th place has the probability of a document belonging to cluster i (before we look
#   at the words)
# words_clusters_probabilities - array, i-th place has a map from words to their probabilities conditioned on cluster i
# yt - the th document (a counter over the document words)
def calc_wti(i, m, z_values):
    if z_values[i] - m < -underflow_k:
        return 0
    else:
        denominator = calc_sum_of_e_power_of_zi(z_values, m)
        numerator = np.power(np.e, z_values[i] - m)
        return numerator / denominator


def calc_sum_of_e_power_of_zi(z_values, m):
    denom_valid_zjs = filter(lambda zj: zj - m >= -underflow_k, z_values)
    denom_list = list(map(lambda zj: np.power(np.e, zj - m), denom_valid_zjs))
    denominator = np.sum(denom_list)
    return denominator


def calc_zis(yt, clusters_probabilities, words_clusters_probabilities):
    n = len(clusters_probabilities)
    z_values = np.zeros(n)
    for j in range(n):
        alpha_j = clusters_probabilities[j]
        zj = np.log(alpha_j) + np.sum(
            [nt_k * np.log(words_clusters_probabilities[word][j]) for word, nt_k in yt.most_common()])
        z_values[j] = zj

    m = np.amax(z_values)
    return z_values, m


def calc_doc_wtis(yt, clusters_probabilities, words_clusters_probabilities):
    z_values, m = calc_zis(yt, clusters_probabilities, words_clusters_probabilities)
    return np.array([calc_wti(i, m, z_values) for i in range(len(clusters_probabilities))])


# returns a list of size N (the number of documents), each list element represents the probability distribution over
# the clusters, so list of lists
def e_step(documents, clusters_probabilities, words_clusters_probabilities):
    return [calc_doc_wtis(yt, clusters_probabilities, words_clusters_probabilities) for yt in documents]


def calc_ai(cluster_idx, wts):
    n = float(len(wts))
    return (1 / n) * np.sum([wt[cluster_idx] for wt in wts])


# word_k - a word
# cluster_idx - the cluster # (0-8)
# documents - list of counters (with word frequencies for each document)
def calc_pik(word_k, cluster_idx, wts, pik_cluster_denominators, documents, vocab_size):
    numerator = np.sum([wts[t][cluster_idx]*document[word_k] for t, document in enumerate(documents)])
    denominator = pik_cluster_denominators[cluster_idx]
    return (numerator + pik_lambda) / (denominator + vocab_size*pik_lambda)


def m_step(wts, documents, vocab, vocab_size):
    cluster_count = len(wts[0])
    clusters_probabilities = np.array([calc_ai(cluster_idx, wts) for cluster_idx in range(cluster_count)])
    clusters_probabilities_with_epsilon_threshold = np.array([c if c > epsilon else epsilon for c in clusters_probabilities])
    clusters_probabilities_sum = np.sum(clusters_probabilities_with_epsilon_threshold)
    smoothed_clusters_probabilities = clusters_probabilities_with_epsilon_threshold / clusters_probabilities_sum

    pik_cluster_denominators = np.array([np.sum([wts[t][cluster_idx] * sum(document.values()) for t, document in enumerate(documents)]) for cluster_idx in range(cluster_count)])
    words_clusters_probabilities = {word_k: np.array([calc_pik(word_k, cluster_idx, wts, pik_cluster_denominators, documents, vocab_size) for cluster_idx in range(cluster_count)]) for word_k in vocab}
    return smoothed_clusters_probabilities, words_clusters_probabilities


def log_likelyhood(documents, clusters_probabilities, words_clusters_probabilities):
    zits = [calc_zis(yt, clusters_probabilities, words_clusters_probabilities) for yt in documents]
    return sum([zit[1] + np.log(calc_sum_of_e_power_of_zi(zit[0], zit[1])) for zit in zits])


def load_input(input_filename):
    with open(input_filename, 'r') as development_set_file:
        lines = development_set_file.readlines()

    vocab_counter = Counter([word for line in lines[1::2] for word in line.strip().split(" ")])
    vocab = set(map(lambda p: p[0], filter(lambda x: x[1] > 3, vocab_counter.most_common())))

    documents_after_filtering = [Counter(list(filter(lambda w: w in vocab, line.strip().split(" ")))) for line in lines[1::2]]

    document_info = lines[0::2]
    documents_info = list(zip(document_info, documents_after_filtering))

    return documents_info, vocab


if __name__ == "__main__":
    documents_info, vocab = load_input("dataset/develop.txt")
    documents = list(map(lambda x: x[1], documents_info))

    # initialize - fake e step where each document gets a cluster by calculating the document index modulo 9
    EYE = np.eye(9)
    wts = [EYE[t % 9] for t in range(len(documents))]

    vocab_size = len(vocab)
    for i in range(10):
        start_time = time()

        clusters_probabilities, words_clusters_probabilities = m_step(wts, documents, vocab, vocab_size)
        wts = e_step(documents, clusters_probabilities, words_clusters_probabilities)

        print(log_likelyhood(documents, clusters_probabilities, words_clusters_probabilities))
        print("took " + str((time() - start_time)) + " seconds")
