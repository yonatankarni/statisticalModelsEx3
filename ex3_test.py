from collections import Counter

from ex3 import calc_wti, calc_ai, e_step, m_step

# initial values - not necessarily completely coherent
clusters_probabilities = [0.4, 0.6]
clusters_word_probabilities = {
    "aaa": [0.2, 0.5],
    "bbb": [0.7, 0.1],
    "ccc": [1, 0.4]
}

documents = [Counter(["aaa", "bbb", "bbb", "ccc", "ccc", "ccc"]), Counter(["aaa", "aaa", "bbb", "ccc"])]

# test calc_wti
y0 = Counter(["aaa", "aaa", "bbb", "bbb"])
wt0_0 = calc_wti(0, y0, clusters_probabilities, clusters_word_probabilities)
wt0_1 = calc_wti(1, y0, clusters_probabilities, clusters_word_probabilities)
print(wt0_0)
print(wt0_1)
assert 0.99999 < wt0_0 + wt0_1 < 1.00001

y1 = Counter(["aaa", "ccc"])
wt1_0 = calc_wti(0, y1, clusters_probabilities, clusters_word_probabilities)
wt1_1 = calc_wti(1, y1, clusters_probabilities, clusters_word_probabilities)
assert 0.99999 < wt1_0 + wt1_1 < 1.00001

wts = [[wt0_0, wt0_1], [wt1_0, wt1_1]]
a0 = calc_ai(0, wts)
a1 = calc_ai(1, wts)
assert 0.99999 < a0 + a1 < 1.00001


# test e-step
clusters_probabilities = [1/9. for x in range(2)]

wts = e_step(documents, clusters_probabilities, clusters_word_probabilities)

vocab = ["aaa", "bbb", "ccc"]
clusters_probabilities, clusters_word_probabilities = m_step(wts, documents, vocab, len(vocab))
