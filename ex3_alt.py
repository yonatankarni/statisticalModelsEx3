import collections
import itertools
import re

import numpy as np


def filter_rare_words_from(articles):
    txt = itertools.chain.from_iterable(article.text for article in articles)
    word_counter = collections.Counter(txt)
    filtered_articles = []
    for article in articles:
        text = article.text
        filtered_text = []
        for word in text:
            if word_counter[word] > 3:
                filtered_text.append(word)
        filtered_articles.append(Article(article.idx, article.topics, filtered_text))
    return filtered_articles


def extract_dictionary_from(articles):
    words = itertools.chain.from_iterable(article.unique_words for article in articles)
    return set(words)


def remove_blank_lines_from(file_obj):
    for line in file_obj:
        line = line.strip()
        if line:
            yield line



class Article:

    def __init__(self, idx, topics, text):
        self.idx = idx
        self.topics = frozenset(topics)
        self.text = text
        self.counter = collections.Counter(text)
        
    
    @property
    def unique_words(self):
        return set(self.counter.keys())
    
    
    def __len__(self):
        return len(self.text)
    
    

    
    def __repr__(self):
        return 'Article({}, {}, {})'.format(repr(self.idx), repr(self.topics), repr(self.text))
    


class Parser:
    
    def __init__(self, file_obj):
        self.document = file_obj
    
    def parse(self):
        lst = list()
        it = remove_blank_lines_from(self.document)
        for i, line in enumerate(it):
            topics = self.parse_header(line)
            text = next(it).split()
            article = Article(i, topics, text)
            lst.append(article)
        
        return lst
        
    def parse_header(self, header):
        pattern = re.compile(r'<TRAIN\s+(\d+)\s+((\w+-?\w+?\s*)+)>')
        m = pattern.match(header)
        if m:
            return m.group(2).split()
        else:
            raise ValueError(header)
        

           

class HistogramMixtureEM:
    
    def __init__(self, articles, dictionary, num_of_clusters):
        self.articles = articles
        self.word_to_idx = dict(zip(dictionary, itertools.count()))
        self.num_of_clusters = num_of_clusters
        self.alpha = None
        self.p = None
        self.clusters = HistogramMixtureEM.init_clusters(articles, num_of_clusters)
        self.w = HistogramMixtureEM.init_w(self.clusters)
        
        
    @staticmethod
    def init_clusters(articles, num_of_clusters):
        groups = itertools.groupby(sorted(articles, key=lambda x: x.idx%num_of_clusters), 
                               key=lambda x: x.idx%num_of_clusters)
        key_to_cluster = collections.OrderedDict()
        for k, group in groups:
            key_to_cluster[k] = list(group)
        assert num_of_clusters == len(key_to_cluster)
    
        return key_to_cluster


    @staticmethod
    def init_w(clusters):
        num_of_clusters = len(clusters)
        num_of_articles = sum(len(cluster) for cluster in clusters.values())
        w = np.zeroes((num_of_articles, num_of_clusters), dtype=np.float64)
        for i, articles in clusters.items():
           for article in articles:
               t = article.idx
               w[t, i] = 1
        return w
    
    def solve(self):
        pass
    
    def e_step(self):
        pass
    
    def m_step(self):
        pass
    
    def compute_alpha(self):
        pass
    
    
    def compute_p(self):
        pass
    
    def compute_w(self):
        for t, doc in enumerate(self.articles):
            z = self.compute_z(doc)
            self.w[t, :] = z / np.sum(z) 
    
    def compute_z(self, doc):
        z = np.zeros_like(self.alpha)
        log_alpha = np.log(self.alpha)
        log_p = np.log(self.p)
        for i, (log_alpha_i, log_pi) in enumerate(zip(log_alpha, log_p)):
            z[i] = self.compute_zi_sum(doc, log_pi) + log_alpha_i
        return z
    
    
    def compute_zi_sum(self, doc, log_pi):
        counter = doc.counter
        s = 0
        for word, count in counter.items():
            if count > 0:
                k = self.word_to_idx[word]
                pik = log_pi[k]
                s += count * pik
        return s
            
    
if __name__ == "__main__":
    with open('./dataset/develop.txt', 'r') as dev_file:
        parser = Parser(dev_file)
        lst = parser.parse()
        assert len(lst) == 2124
    assert len(lst) == len(filter_rare_words_from(lst))
    num_of_clusters = 9
    dictionary = extract_dictionary_from(lst)