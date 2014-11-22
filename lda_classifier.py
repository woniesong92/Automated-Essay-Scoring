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


if __name__ == "__main__":
    myParser = parser.Parser()
    training_examples = myParser.parse("data/training_set_rel3.tsv")


    # training_examples = list(filter(lambda x: x["essay_set"]== "8", training_examples))


    

    myFeatureExtractor = feature_extractor.FeatureExtractor()
    essays = myFeatureExtractor.extract_essay_and_scores(training_examples)[0]
    new_essays = [tokenize_no_stop_words(essay) for essay in essays]


    pdb.set_trace()

    texts = new_essays



    myDict = gensim.corpora.dictionary.Dictionary(new_essays)


    corpus = [myDict.doc2bow(text) for text in texts]

    pdb.set_trace()


    ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics = 5)

    print "DONEess"
    pdb.set_trace()




