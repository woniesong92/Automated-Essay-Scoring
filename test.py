from termcolor import colored
import parser
import feature_extractor
import pickle
import re
import math
import pdb
import nltk
import logging, gensim, bz2
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import KNeighborsRegressor as knnR
from sklearn.svm import SVC as svm
from sklearn.svm import SVR as svr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np


###########################
#       LOAD METHODS      #
###########################
# returns tfidf feature vectors in essay set i
# data_type: train, test
def get_tfidf(i, data_type):
    features = pickle.load(open("data/set%d_%s_tfidf_matrix.pkl" % (i,data_type), 'r'))
    return features

# returns normalized tfidf feature vectors in essay set i
def get_normalized_tfidf(i, data_type):
    features = pickle.load(open("data/set%d_%s_normalized_tfidf_matrix.pkl" % (i,data_type), 'r'))
    return features

# returns extra feature vectors of in essay set i
def get_extra_features(i, data_type):
    features = pickle.load(open("data/set%d_%s_statistics.pkl" % (i,data_type), 'r'))
    return features

# returns extra feature vectors of in essay set i
def get_normalized_extra_features(i, data_type):
    features = pickle.load(open("data/set%d_%s_normalized_statistics.pkl" % (i,data_type), 'r'))
    return features

# returns LDA model with iterations=iterations and passes=passes for essay set i
def get_lda(i, iterations, passes, num_topics_for_lda):
    # prepare dictionary and corpus
    myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % i)
    corpus = gensim.corpora.MmCorpus('data/set%d.mm' % i)

    lda = gensim.models.LdaModel(corpus=corpus, id2word=myDict, num_topics=num_topics_for_lda, iterations=iterations, passes=passes)

    topic_vectors = []
    for doc in corpus:
        topic_distribution = lda[doc]
        topic_vector = topic_distribution_to_vector(topic_distribution, num_topics_for_lda)
        topic_vectors.append(topic_vector)

    return (lda, topic_vectors)

# returns (test_essays, test_scores) in essay set i
def get_test_examples(i):
    test_examples = parser.Parser().parse("data/set%d_test.tsv" % i)
    test_essays, test_scores = feature_extractor.FeatureExtractor().extract_essay_and_scores(test_examples) #list of raw essays and their scores
    return ([tokenize_no_stop_words(essay) for essay in test_essays], test_scores)


#############################
#       HELPER METHODS      #
#############################

def tokenize_no_stop_words(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    f = open('stopwords.txt')
    stoplist = set(line.split('\n')[0] for line in f)
    filtered_words = [w for w in tokens if not w in stoplist]
    return filtered_words

def topic_distribution_to_vector(topic_distribution, num_topics):
    vector = [0 for x in range(num_topics)]
    for (topic_id, prob) in topic_distribution:
        vector[topic_id] = prob
    return vector

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

    return feature_selection.get_support()


###########################
#       TEST METHODS      #
###########################
def combined_test(i):
    # Get feature vectors
    tfidf_vectors = get_tfidf(i, 'train')
    extra_features_vector = get_extra_features(i, 'train')
    num_topics_for_lda = 30
    lda,lda_vector = get_lda(i, 100, 1, num_topics_for_lda)
    train_features = np.concatenate((tfidf_vectors,extra_features_vector),1)
    train_features = np.concatenate((lda_vector,train_features),1)
    
    normalized_tfidf_vectors = get_normalized_tfidf(i, 'train')
    normalized_extra_features_vector = get_normalized_extra_features(i, 'train')
    normalized_train_features = np.concatenate((normalized_tfidf_vectors,normalized_extra_features_vector),1)
    normalized_train_features = np.concatenate((lda_vector,normalized_train_features),1)   

    tfidf_train_features = tfidf_vectors
    extra_features_train_features = extra_features_vector
    lda_train_features = lda_vector

    print colored('feature vectors loaded', 'cyan')

    # Set up classifiers
    neigh = knn(n_neighbors=5, weights = 'distance')
    svm_classifier = svm()
    svr_classifier = svr()
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    normalized_neigh = knn(n_neighbors=5, weights = 'distance')
    normalized_svm_classifier = svm()
    normalized_svr_classifier = svr()
    normalized_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    tfidf_neigh = knn(n_neighbors=5, weights = 'distance')
    tfidf_svm_classifier = svm()
    tfidf_svr_classifier = svr()
    tfidf_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    extra_features_neigh = knn(n_neighbors=5, weights = 'distance')
    extra_features_svm_classifier = svm()
    extra_features_svr_classifier = svr()
    extra_features_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    lda_neigh = knn(n_neighbors=5, weights = 'distance')
    lda_svm_classifier = svm()
    lda_svr_classifier = svr()
    lda_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    # Load training essay dictionary and corpus
    myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % i)
    corpus = gensim.corpora.MmCorpus('data/set%d.mm' % i)

    # train classifiers
    neigh.fit(train_features, scores)
    svm_classifier.fit(train_features, scores)
    knnr_classifier.fit(train_features, scores)

    normalized_neigh.fit(normalized_train_features, scores)
    normalized_svm_classifier.fit(normalized_train_features, scores)
    normalized_knnr_classifier.fit(normalized_train_features, scores)

    tfidf_neigh.fit(tfidf_train_features, scores)
    tfidf_svm_classifier.fit(tfidf_train_features, scores)
    tfidf_knnr_classifier.fit(tfidf_train_features, scores)

    extra_features_neigh.fit(extra_features_train_features, scores)
    extra_features_svm_classifier.fit(extra_features_train_features, scores)
    extra_features_knnr_classifier.fit(extra_features_train_features, scores)

    lda_neigh.fit(lda_train_features, scores)
    lda_svm_classifier.fit(lda_train_features, scores)
    lda_knnr_classifier.fit(lda_train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    num_correct_neigh = 0
    num_correct_svm = 0
    num_correct_svr = 0

    num_correct_normalized_neigh = 0
    num_correct_normalized_svm = 0
    num_correct_normalized_svr = 0

    num_correct_tfidf_neigh = 0
    num_correct_tfidf_svm = 0
    num_correct_tfidf_svr = 0

    num_correct_extra_features_neigh = 0
    num_correct_extra_features_svm = 0
    num_correct_extra_features_svr = 0

    num_correct_lda_neigh = 0
    num_correct_lda_svm = 0
    num_correct_lda_svr = 0

    index = 0

    # Load test essay feature vectors
    tfidf_test = get_tfidf(i, 'test')
    extra_features_test = get_extra_features(i, 'test')
    test_features = np.concatenate((tfidf_test,extra_features_test),1)

    normalized_tfidf_test = get_normalized_tfidf(i, 'test')
    normalized_extra_features_test = get_normalized_extra_features(i, 'test')
    normalized_test_features = np.concatenate((normalized_tfidf_test,normalized_extra_features_test),1)


    predicted = []
    normalized_predicted = []
    tfidf_predicted = []
    extra_features_predicted = []
    lda_predicted = []

    actual = []

    print colored('Testing...', 'cyan')

    for idx, test_essay in enumerate(test_essays):
        doc_bow = myDict.doc2bow(test_essay)
        doc_lda = lda[doc_bow]
        
        # Test feature vectors
        vectorized_lda = topic_distribution_to_vector(doc_lda, num_topics_for_lda)
        test_feature = np.concatenate((vectorized_lda, test_features[index]), 1)        
        normalized_test_feature = np.concatenate((vectorized_lda, normalized_test_features[index]), 1)
        tfidf_test_feature = tfidf_test[index]
        extra_features_test_feature = extra_features_test[index]
        lda_test_feature = vectorized_lda

        predicted_score = neigh.predict(test_feature)
        noramlized_predicted_score = noramlized_neigh.predict(normalized_test_feature)
        tfidf_predicted_score = tfidf_neigh.predict(tfidf_test_feature)
        extra_features_predicted_score = extra_features_neigh.predict(extra_features_test_feature)
        lda_predicted_score = lda_neigh.predict(lda_test_feature)


    print colored('ESSAY SET %d KNN ACCURACY: %f' % (i, 100*accuracy_knn), 'green', attrs=['bold'])
    print colored('ESSAY SET %d SVM ACCURACY: %f' % (i, 100*accuracy_svm), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])



def test(i):
    # Get feature vectors
    tfidf_vectors = get_tfidf(i, 'train')
    extra_features_vector = get_extra_features(i, 'train')
    num_topics_for_lda = 30
    lda,lda_vector = get_lda(i, 100, 1, num_topics_for_lda)
    train_features = np.concatenate((tfidf_vectors,extra_features_vector),1)
    train_features = np.concatenate((lda_vector,train_features),1)
    print colored('feature vectors loaded', 'cyan')

    # Set up classifiers
    neigh = knn(n_neighbors=5, weights = 'distance')
    svm_classifier = svm()
    svr_classifier = svr()
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    # Load training essay dictionary and corpus
    myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % i)
    corpus = gensim.corpora.MmCorpus('data/set%d.mm' % i)

    neigh.fit(train_features, scores)
    svm_classifier.fit(train_features, scores)
    knnr_classifier.fit(train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    num_correct_neigh = 0
    num_correct_svm = 0
    num_correct_svr = 0

    index = 0

    # Load test essay feature vectors
    tfidf_test = get_tfidf(i, 'test')
    extra_features_test = get_extra_features(i, 'test')
    test_features = np.concatenate((tfidf_test,extra_features_test),1)

    predicted = []
    actual = []
    print colored('Testing...', 'cyan')
    for idx, test_essay in enumerate(test_essays):
        doc_bow = myDict.doc2bow(test_essay)
        doc_lda = lda[doc_bow]
        
        vectorized_lda = topic_distribution_to_vector(doc_lda, num_topics_for_lda)
        test_feature = np.concatenate((vectorized_lda, test_features[index]), 1)        

        predicted_score = neigh.predict(test_feature)
        # print "Actual:", test_scores[idx], "Predicted:", predicted_score
        if abs(int(test_scores[idx]) - int(predicted_score[0])) <= 1:
            num_correct_neigh += 1

        predicted_score2 = svm_classifier.predict(test_feature)
        if abs(int(test_scores[idx]) - int(predicted_score2[0])) <= 1:
            num_correct_svm += 1

        print "Actual:", test_scores[idx], "Predicted:", predicted_score

        predicted.append(float(knnr_classifier.predict(test_feature)))
        actual.append(float(test_scores[idx]))
        index += 1

    accuracy_knn = num_correct_neigh / float(len(test_scores))
    accuracy_svm = num_correct_svm / float(len(test_scores))

    print colored('ESSAY SET %d KNN ACCURACY: %f' % (i, 100*accuracy_knn), 'green', attrs=['bold'])
    print colored('ESSAY SET %d SVM ACCURACY: %f' % (i, 100*accuracy_svm), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])

def normalized_test(i):
    # Get feature vectors
    tfidf_vectors = get_normalized_tfidf(i, 'train')
    extra_features_vector = get_normalized_extra_features(i, 'train')
    num_topics_for_lda = 30
    lda,lda_vector = get_lda(i, 100, 1, num_topics_for_lda)
    train_features = np.concatenate((tfidf_vectors,extra_features_vector),1)
    train_features = np.concatenate((lda_vector,train_features),1)
    print colored('feature vectors loaded', 'cyan')

    # Set up classifiers
    neigh = knn(n_neighbors=5, weights = 'distance')
    svm_classifier = svm()
    svr_classifier = svr()
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    # Load training essay dictionary and corpus
    myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % i)
    corpus = gensim.corpora.MmCorpus('data/set%d.mm' % i)

    neigh.fit(train_features, scores)
    svm_classifier.fit(train_features, scores)
    knnr_classifier.fit(train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    num_correct_neigh = 0
    num_correct_svm = 0
    num_correct_svr = 0

    index = 0

    # Load test essay feature vectors
    tfidf_test = get_tfidf(i, 'test')
    extra_features_test = get_extra_features(i, 'test')
    test_features = np.concatenate((tfidf_test,extra_features_test),1)

    predicted = []
    actual = []
    print colored('Testing...', 'cyan')
    for idx, test_essay in enumerate(test_essays):
        doc_bow = myDict.doc2bow(test_essay)
        doc_lda = lda[doc_bow]
        
        vectorized_lda = topic_distribution_to_vector(doc_lda, num_topics_for_lda)
        test_feature = np.concatenate((vectorized_lda, test_features[index]), 1)        

        predicted_score = neigh.predict(test_feature)
        # print "Actual:", test_scores[idx], "Predicted:", predicted_score
        if abs(int(test_scores[idx]) - int(predicted_score[0])) <= 1:
            num_correct_neigh += 1

        predicted_score2 = svm_classifier.predict(test_feature)
        if abs(int(test_scores[idx]) - int(predicted_score2[0])) <= 1:
            num_correct_svm += 1

        print "Actual:", test_scores[idx], "Predicted:", predicted_score

        predicted.append(float(knnr_classifier.predict(test_feature)))
        actual.append(float(test_scores[idx]))
        index += 1

    accuracy_knn = num_correct_neigh / float(len(test_scores))
    accuracy_svm = num_correct_svm / float(len(test_scores))

    print colored('ESSAY SET %d KNN ACCURACY: %f' % (i, 100*accuracy_knn), 'green', attrs=['bold'])
    print colored('ESSAY SET %d SVM ACCURACY: %f' % (i, 100*accuracy_svm), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])


def tfidf_test(i):
    # Get feature vectors
    train_features = get_tfidf(i, 'train')

    # Set up classifiers
    neigh = knn(n_neighbors=5, weights = 'distance')
    svm_classifier = svm()
    svr_classifier = svr()
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    neigh.fit(train_features, scores)
    svm_classifier.fit(train_features, scores)
    knnr_classifier.fit(train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    num_correct_neigh = 0
    num_correct_svm = 0
    num_correct_svr = 0

    index = 0
    test_features = get_tfidf(i, 'test')

    predicted = []
    actual = []

    for idx, test_essay in enumerate(test_essays):
        test_feature = test_features[index]   

        predicted_score = neigh.predict(test_feature)
        # print "Actual:", test_scores[idx], "Predicted:", predicted_score
        if abs(int(test_scores[idx]) - int(predicted_score[0])) <= 1:
            num_correct_neigh += 1

        predicted_score2 = svm_classifier.predict(test_feature)
        if abs(int(test_scores[idx]) - int(predicted_score2[0])) <= 1:
            num_correct_svm += 1

        print "Actual:", test_scores[idx], "Predicted:", predicted_score

        predicted.append(float(knnr_classifier.predict(test_feature)))
        actual.append(float(test_scores[idx]))

        index += 1

    accuracy_knn = num_correct_neigh / float(len(test_scores))
    accuracy_svm = num_correct_svm / float(len(test_scores))

    print colored('ESSAY SET %d KNN ACCURACY: %f' % (i, 100*accuracy_knn), 'green', attrs=['bold'])
    print colored('ESSAY SET %d SVM ACCURACY: %f' % (i, 100*accuracy_svm), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])

def extra_features_test(i):
    # Get feature vectors
    train_features = get_extra_features(i, 'train')

    # Set up classifiers
    neigh = knn(n_neighbors=5, weights = 'distance')
    svm_classifier = svm()
    svr_classifier = svr()
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    neigh.fit(train_features, scores)
    svm_classifier.fit(train_features, scores)
    knnr_classifier.fit(train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    num_correct_neigh = 0
    num_correct_svm = 0
    num_correct_svr = 0

    index = 0
    test_features = get_extra_features(i, 'test')

    predicted = []
    actual = []

    for idx, test_essay in enumerate(test_essays):
        test_feature = test_features[index]   

        predicted_score = neigh.predict(test_feature)
        # print "Actual:", test_scores[idx], "Predicted:", predicted_score
        if abs(int(test_scores[idx]) - int(predicted_score[0])) <= 1:
            num_correct_neigh += 1

        predicted_score2 = svm_classifier.predict(test_feature)
        if abs(int(test_scores[idx]) - int(predicted_score2[0])) <= 1:
            num_correct_svm += 1

        print "Actual:", test_scores[idx], "Predicted:", predicted_score

        predicted.append(float(knnr_classifier.predict(test_feature)))
        actual.append(float(test_scores[idx]))

        index += 1

    accuracy_knn = num_correct_neigh / float(len(test_scores))
    accuracy_svm = num_correct_svm / float(len(test_scores))

    print colored('ESSAY SET %d KNN ACCURACY: %f' % (i, 100*accuracy_knn), 'green', attrs=['bold'])
    print colored('ESSAY SET %d SVM ACCURACY: %f' % (i, 100*accuracy_svm), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])

def lda_test(i):
    # Get feature vectors
    num_topics_for_lda = 30
    lda,lda_vector = get_lda(i, 100, 1, num_topics_for_lda)
    train_features = lda_vector
    print colored('feature vectors loaded', 'cyan')

    # Set up classifiers
    neigh = knn(n_neighbors=5, weights = 'distance')
    svm_classifier = svm()
    svr_classifier = svr()
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    # Load training essay dictionary and corpus
    myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % i)
    corpus = gensim.corpora.MmCorpus('data/set%d.mm' % i)

    neigh.fit(train_features, scores)
    svm_classifier.fit(train_features, scores)
    knnr_classifier.fit(train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    num_correct_neigh = 0
    num_correct_svm = 0
    num_correct_svr = 0

    index = 0

    predicted = []
    actual = []

    for idx, test_essay in enumerate(test_essays):
        doc_bow = myDict.doc2bow(test_essay)
        doc_lda = lda[doc_bow]
        
        test_feature = topic_distribution_to_vector(doc_lda, num_topics_for_lda)    

        predicted_score = neigh.predict(test_feature)
        # print "Actual:", test_scores[idx], "Predicted:", predicted_score
        if abs(int(test_scores[idx]) - int(predicted_score[0])) <= 1:
            num_correct_neigh += 1

        predicted_score2 = svm_classifier.predict(test_feature)
        if abs(int(test_scores[idx]) - int(predicted_score2[0])) <= 1:
            num_correct_svm += 1

        print "Actual:", test_scores[idx], "Predicted:", predicted_score

        predicted.append(float(knnr_classifier.predict(test_feature)))
        actual.append(float(test_scores[idx]))

        index += 1

    accuracy_knn = num_correct_neigh / float(len(test_scores))
    accuracy_svm = num_correct_svm / float(len(test_scores))

    print colored('ESSAY SET %d KNN ACCURACY: %f' % (i, 100*accuracy_knn), 'green', attrs=['bold'])
    print colored('ESSAY SET %d SVM ACCURACY: %f' % (i, 100*accuracy_svm), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
    print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (i, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])


###################
#       MAIN      #
###################

def main():
    essay_sets = [1,2,3,4,5,6,7,8]

    for essay_set in essay_sets:
        test(essay_set)

if __name__ == "__main__":
    main()      