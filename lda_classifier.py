import parser
import pdb
import numpy as np
import pickle
import nltk
import feature_extractor

import gensim.models.ldamodel
import gensim.corpora.dictionary 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import logging, bz2
from sklearn.feature_extraction import DictVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stemmer = PorterStemmer()

f = open('stopwords.txt')
stoplist = set(line.split('\n')[0] for line in f)


def essays_tokenized(essays):
    pass

def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

def topic_distribution_to_vector(topic_distribution, num_topics):
    vector = [0 for x in range(num_topics)]
    for (topic_id, prob) in topic_distribution:
        vector[topic_id] = prob
    return vector

#with stemming
# def tokenize_no_stop_words(text):
#     tokens = nltk.word_tokenize(text)
#     stems = stem_tokens(tokens, stemmer)
#     filtered_words = [w for w in stems if not w in stoplist]
#     return filtered_words

#FIXME: without stemming
def tokenize_no_stop_words(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    filtered_words = [w for w in tokens if not w in stoplist]

    return filtered_words


def get_best_features(corpus, scores, num_features):
    new_corpus = []
    for document in corpus:
        new_document = {}
        for (word, freq) in document:
            new_document[word] = freq
        new_corpus.append(new_document)
    v = DictVectorizer()
    X = v.fit_transform(new_corpus)
    y = scores
    # pdb.set_trace()
    feature_selection = SelectKBest(chi2, k=num_features).fit(X, y)
    new_X = SelectKBest(chi2, k=num_features).fit(X, y)

    return (gensim.matutils.Sparse2Corpus(new_X.transform(X)), y, feature_selection.get_support())


if __name__ == "__main__":

    myParser = parser.Parser()
    training_examples = myParser.parse("data/set1_train.tsv")
    # training_examples = list(filter(lambda x: x["essay_set"]== "8", training_examples))
    myFeatureExtractor = feature_extractor.FeatureExtractor()
    essays, scores = myFeatureExtractor.extract_essay_and_scores(training_examples) #list of raw essays and their scores
    new_essays = [tokenize_no_stop_words(essay) for essay in essays]

    texts = new_essays # essays without stopping words

    myDict = gensim.corpora.dictionary.Dictionary(new_essays)

    myDict.filter_extremes(no_below=1, no_above=0.1, keep_n=100000)

    corpus = [myDict.doc2bow(text) for text in texts]

    corpus_old = corpus

    #corpus, scores, features = get_best_features(corpus, scores, 'all') #feature selected corpus

    # for f in range(len(features)):
    #     if features[f]:
    #         print myDict.get(f)
    #pdb.set_trace()

    lda_50 = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics = 20, iterations=1000, passes=50)
    lda_100 = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics = 20, iterations=1000, passes=100)
    lda_250 = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics = 20, iterations=1000, passes=250)
    lda_500 = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics = 20, iterations=1000, passes=500)
    lda_1000 = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics = 20, iterations=1000, passes=1000)

    pickle.dump(lda_50, open("ldamodel_50passes.p", "wb"))
    pickle.dump(lda_100, open("ldamodel_100passes.p", "wb"))
    pickle.dump(lda_250, open("ldamodel_250passes.p", "wb"))
    pickle.dump(lda_500, open("ldamodel_500passes.p", "wb"))
    pickle.dump(lda_1000, open("ldamodel_1000passes.p", "wb"))

    print "======= DONE DONE DONE DONE DONE DONE DONE DONE ======"
    #pdb.set_trace()




