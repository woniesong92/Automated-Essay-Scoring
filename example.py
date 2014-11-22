



import parser
import pdb
import numpy as np
import pickle
import nltk
import feature_extractor

import gensim.models.ldamodel
import gensim.corpora.dictionary 

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()


def essays_tokenized(essays):
    pass


def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

def tokenize_no_stop_words(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    filtered_words = [w for w in stems if not w in stopwords.words('english')]
    return filtered_words




essay = ""
with open("essay.txt") as essay_file:
    for line in essay_file:
        essay += line

pdb.set_trace()
