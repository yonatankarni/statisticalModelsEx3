from collections import Counter

from ex3 import calc_wti, calc_ai, e_step

# test calc_wti
clusters_probabilities = [0.4, 0.6]
clusters_word_probabilities = [
    {"aaa": 0.2, "bbb": 0.8, "ccc": 0},
    {"aaa": 0.5, "bbb": 0.1, "ccc": 0.4}
]
y0 = Counter(["aaa", "aaa", "bbb", "bbb"])
wt0_0 = calc_wti(0, clusters_probabilities, clusters_word_probabilities, y0)
wt0_1 = calc_wti(1, clusters_probabilities, clusters_word_probabilities, y0)
assert 0.99999 < wt0_0 + wt0_1 < 1.00001

y1 = Counter(["aaa", "ccc"])
wt1_0 = calc_wti(0, clusters_probabilities, clusters_word_probabilities, y1)
wt1_1 = calc_wti(1, clusters_probabilities, clusters_word_probabilities, y1)
assert 0.99999 < wt1_0 + wt1_1 < 1.00001

wts = [[wt0_0, wt0_1], [wt1_0, wt1_1]]
a0 = calc_ai(0, wts)
a1 = calc_ai(1, wts)
assert 0.99999 < a0 + a1 < 1.00001