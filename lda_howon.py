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

from sklearn.neighbors import KNeighborsClassifier as knn

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
def tokenize_no_stop_words(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    filtered_words = [w for w in stems if not w in stoplist]
    return filtered_words

#FIXME: without stemming
# def tokenize_no_stop_words(text):
#     tokens = nltk.word_tokenize(text)
#     stems = stem_tokens(tokens, stemmer)
#     filtered_words = [w for w in tokens if not w in stoplist]

#     return filtered_words


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
    pdb.set_trace()
    feature_selection = SelectKBest(chi2, k=num_features).fit(X, y)
    new_X = SelectKBest(chi2, k=num_features).fit(X, y)

    return feature_selection.get_support()


if __name__ == "__main__":

    myParser = parser.Parser()
    training_examples = myParser.parse("data/set1_train.tsv")
    # training_examples = list(filter(lambda x: x["essay_set"]== "8", training_examples))
    myFeatureExtractor = feature_extractor.FeatureExtractor()
    essays, scores = myFeatureExtractor.extract_essay_and_scores(training_examples) #list of raw essays and their scores
    new_essays = [tokenize_no_stop_words(essay) for essay in essays]

    texts = new_essays # essays without stopping words

    myDict = gensim.corpora.dictionary.Dictionary(new_essays)

    # myDict.filter_extremes(no_below=1, no_above=0.1, keep_n=100000) # FIXME: to include or not?

    corpus = [myDict.doc2bow(text) for text in texts]

    corpus_old = corpus

    # pdb.set_trace()

    bad_good_features = get_best_features(corpus, scores, 10093-100) #feature selected corpus
    bad_ids = []
    for i in range(len(bad_good_features)):
        if not bad_good_features[i]:
            bad_ids.append(bad_good_features[i])
    myDict.filter_tokens(bad_ids=bad_ids)


    # for f in range(len(features)):
    #     if features[f]:
    #         print myDict.get(f)
    #pdb.set_trace()

    corpus = [myDict.doc2bow(text) for text in texts]
    num_topics_for_lda = 20
    lda = gensim.models.ldamodel.LdaModel(corpus, id2word = myDict, num_topics=num_topics_for_lda, iterations=1000, passes=10)
    topic_vectors = []
    for doc in corpus:
        topic_distribution = lda[doc]
        topic_vector = topic_distribution_to_vector(topic_distribution, num_topics_for_lda)
        topic_vectors.append(topic_vector)
    print "======= PREPARED TOPIC VECTORS ======"
    
    neigh = knn(n_neighbors=5)
    pdb.set_trace()
    neigh.fit(topic_vectors, scores)

    print "======= PREPARED KNN, NOW LOAD TEST SET ======"

    test_examples = myParser.parse("data/set1_test.tsv")
    test_essays, test_scores = myFeatureExtractor.extract_essay_and_scores(test_examples) #list of raw essays and their scores
    new_test_essays = [tokenize_no_stop_words(essay) for essay in test_essays]

    #myDict.add_documents(new_test_essays)

    num_correct = 0

    for idx, test_essay in enumerate(new_test_essays):
        doc_bow = myDict.doc2bow(test_essay)
        doc_lda = lda[doc_bow]
        vectorized_lda = topic_distribution_to_vector(doc_lda, num_topics_for_lda)
        predicted_score = neigh.predict(vectorized_lda)
        print "Actual:", test_scores[idx], "Predicted:", predicted_score
        if int(test_scores[idx]) == int(predicted_score[0]):
            num_correct += 1

    print "======= DONE DONE DONE ======"

    accuracy = num_correct / float(len(test_scores))
    print "ACCURACY:", accuracy
