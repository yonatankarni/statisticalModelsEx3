from collections import Counter
import numpy as np


# calc_wti: calculate the probability for cluster i given doc t
# i: cluster index (0..8)
# clusters_probabilities - array, i-th place has the probability of a document belonging to cluster i (before we look
#   at the words)
# clusters_word_probabilities - array, i-th place has a map from words to their probabilities conditioned on cluster i
# yt - the th document (a counter over the document words)
def calc_wti(i, clusters_probabilities, clusters_word_probabilities, yt):
    n = len(clusters_probabilities)
    alpha_i = clusters_probabilities[i]
    document_piks = [pow(clusters_word_probabilities[i][word], cnt) for word, cnt in yt.most_common()]
    numerator = alpha_i * np.prod(document_piks)
    denom_list = np.zeros(n)
    for j in range(n):
        alpha_j = clusters_probabilities[j]
        cluster_document_piks = [pow(clusters_word_probabilities[j][word], cnt) for word, cnt in yt.most_common()]
        denom_list[j] = alpha_j * np.prod(cluster_document_piks)
    denominator = np.sum(denom_list)

    return numerator / denominator


# returns a list of size N (the number of documents), each list element represents the probability distribution over
# the clusters, so list of lists
def e_step(documents, clusters_probabilities, clusters_word_probabilities):
    result = [calc_wti(t, clusters_probabilities, clusters_word_probabilities, doc) for t, doc in documents.items()]
    return result


def calc_ai(cluster_idx, wts):
    n = float(len(wts))
    return (1 / n) * np.sum([wt[cluster_idx] for wt in wts])


# word_k - a word
# cluster_idx - the cluster # (0-8)
# documents - list of counters (with word frequencies for each document)
def calc_pik(word_k, cluster_idx, wts, documents):
    numerator = np.sum([wts[t][cluster_idx]*document[word_k] for t, document in documents.items()])
    denominator = np.sum([wts[t][cluster_idx]*sum(document.values()) for t, document in documents.items()])
    return numerator / denominator


def m_step():
    pass


def load_input(input_filename):
    with open(input_filename, 'r') as development_set_file:
        lines = development_set_file.readlines()

    zipped = zip(lines[0::2], lines[1::2])

    res = map(lambda x: (x[0], x[1].strip().split(" "), Counter(x[1].strip().split(" "))), zipped)

    return res


if __name__ == "__main__":
    res = load_input("dataset/develop.txt")

    print(res[0])
