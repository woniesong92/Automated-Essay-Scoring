from termcolor import colored
import parser
import feature_extractor
import pickle
import re
import math
import pdb
import nltk
import enchant
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


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
    return ([tokenize(essay) for essay in test_essays], test_scores)


#############################
#       HELPER METHODS      #
#############################

def remove_symbols(essay):
    return re.sub(r'[^\w]', '', essay.lower())

def tokenize(text):
    tokenized_text = []

    f = open('stopwords.txt')
    dt = enchant.Dict("en_US")
    stoplist = set(line.split('\n')[0] for line in f)

    for word in text.lower().split():
        word = remove_symbols(word)
        if word != '':
            if dt.check(word) and word not in stoplist:
                tokenized_text.append(word)
            elif not dt.check(word):
                suggestions = dt.suggest(word)
                if len(suggestions) > 0:
                    word = suggestions[0]
                    if word not in stoplist:
                        tokenized_text.append(word)

    return tokenized_text


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
    feature_selection = SelectKBest(chi2, k=num_features).fit(X, y)
    new_X = SelectKBest(chi2, k=num_features).fit(X, y)

    return feature_selection.get_support()

def filter_nan(actual, predicted):
    filtered_actual = []
    filtered_predicted = [] 

    for a,p in zip(actual,predicted):
        if not math.isnan(p):
            filtered_actual.append(a)
            filtered_predicted.append(p)

    return (filtered_actual, filtered_predicted)


###########################
#       TEST METHODS      #
###########################
def combined_test(i):
    # Get feature vectors
    tfidf_vectors = get_tfidf(i, 'train')
    extra_features_vector = get_extra_features(i, 'train')
    num_topics_for_lda = 30
    lda,lda_vector = get_lda(i, 100, 10, num_topics_for_lda)
    train_features = np.concatenate((tfidf_vectors,extra_features_vector),1)
    train_features = np.concatenate((lda_vector,train_features),1)
    
    normalized_tfidf_vectors = get_normalized_tfidf(i, 'train')
    normalized_extra_features_vector = get_normalized_extra_features(i, 'train')
    normalized_train_features = np.concatenate((normalized_tfidf_vectors,normalized_extra_features_vector),1)
    normalized_train_features = np.concatenate((lda_vector,normalized_train_features),1)   

    tfidf_train_features = tfidf_vectors
    extra_features_train_features = extra_features_vector
    lda_train_features = lda_vector
    tfidf_extra_train_features = np.concatenate((tfidf_vectors,extra_features_vector),1)
    lda_extra_train_features = np.concatenate((lda_vector,extra_features_vector),1)

    print colored('feature vectors loaded', 'cyan')

    # Set up classifiers
    knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    svr_classifier = svr()

    normalized_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    normalized_svr_classifier = svr()

    tfidf_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    tfidf_svr_classifier = svr()

    extra_features_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    extra_features_svr_classifier = svr()

    lda_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    lda_svr_classifier = svr()

    tfidf_extra_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    tfidf_extra_svr_classifier = svr()

    lda_extra_knnr_classifier = knnR(n_neighbors=5, weights = 'distance')
    lda_extra_svr_classifier = svr()


    print colored('classifiers setup', 'cyan')

    # Load training essay scores
    scores = []
    with open('data/set%d.scores' % i) as f:
        for score in f:
            scores.append(int(score.split('\n')[0]))

    # Load training essay dictionary and corpus
    myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % i)
    corpus = gensim.corpora.MmCorpus('data/set%d.mm' % i)

    # train classifiers
    knnr_classifier.fit(train_features, scores)
    svr_classifier.fit(train_features, scores)

    normalized_knnr_classifier.fit(normalized_train_features, scores)
    normalized_svr_classifier.fit(normalized_train_features, scores)

    tfidf_knnr_classifier.fit(tfidf_train_features, scores)
    tfidf_svr_classifier.fit(tfidf_train_features, scores)

    extra_features_knnr_classifier.fit(extra_features_train_features, scores)
    extra_features_svr_classifier.fit(extra_features_train_features, scores)

    lda_knnr_classifier.fit(lda_train_features, scores)
    lda_svr_classifier.fit(lda_train_features, scores)

    tfidf_extra_knnr_classifier.fit(tfidf_extra_train_features, scores)
    tfidf_extra_svr_classifier.fit(tfidf_extra_train_features, scores)

    lda_extra_knnr_classifier.fit(lda_extra_train_features, scores)
    lda_extra_svr_classifier.fit(lda_extra_train_features, scores)

    test_essays,test_scores = get_test_examples(i)

    index = 0

    print colored('classifiers trained', 'cyan')

    # Load test essay feature vectors
    tfidf_test = get_tfidf(i, 'test')
    extra_features_test = get_extra_features(i, 'test')
    test_features = np.concatenate((tfidf_test,extra_features_test),1)

    normalized_tfidf_test = get_normalized_tfidf(i, 'test')
    normalized_extra_features_test = get_normalized_extra_features(i, 'test')
    normalized_test_features = np.concatenate((normalized_tfidf_test,normalized_extra_features_test),1)


    knnr_predicted = []
    svr_predicted = []
    knnr_normalized_predicted = []
    svr_normalized_predicted = []
    knnr_tfidf_predicted = []
    svr_tfidf_predicted = []
    knnr_extra_features_predicted = []
    svr_extra_features_predicted = []
    knnr_lda_predicted = []
    svr_lda_predicted = []
    knnr_tfidf_extra_predicted = []
    svr_tfidf_extra_predicted = []
    knnr_lda_extra_predicted = []
    svr_lda_extra_predicted = []

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
        tfidf_extra_feature = test_features[index]
        lda_extra_feature = np.concatenate((vectorized_lda,extra_features_test[index]), 1)

        knnr_predicted_score = knnr_classifier.predict(test_feature)
        svr_predicted_score = svr_classifier.predict(test_feature)
        knnr_normalized_predicted_score = normalized_knnr_classifier.predict(normalized_test_feature)
        svr_normalized_predicted_score = normalized_svr_classifier.predict(normalized_test_feature)
        knnr_tfidf_predicted_score = tfidf_knnr_classifier.predict(tfidf_test_feature)
        svr_tfidf_predicted_score = tfidf_svr_classifier.predict(tfidf_test_feature)
        knnr_extra_features_predicted_score = extra_features_knnr_classifier.predict(extra_features_test_feature)
        svr_extra_features_predicted_score = extra_features_svr_classifier.predict(extra_features_test_feature)
        knnr_lda_predicted_score = lda_knnr_classifier.predict(lda_test_feature)
        svr_lda_predicted_score = lda_svr_classifier.predict(lda_test_feature)        
        knnr_tfidf_extra_predicted_score = tfidf_extra_knnr_classifier.predict(tfidf_extra_feature)
        svr_tfidf_extra_predicted_score = tfidf_extra_svr_classifier.predict(tfidf_extra_feature)
        knnr_lda_extra_predicted_score = lda_extra_knnr_classifier.predict(lda_extra_feature)
        svr_lda_extra_predicted_score = lda_extra_svr_classifier.predict(lda_extra_feature)

        actual.append(float(test_scores[idx]))
        knnr_predicted.append(float(knnr_predicted_score))
        svr_predicted.append(float(svr_predicted_score))
        knnr_normalized_predicted.append(float(knnr_normalized_predicted_score))
        svr_normalized_predicted.append(float(svr_normalized_predicted_score))
        knnr_tfidf_predicted.append(float(knnr_tfidf_predicted_score))
        svr_tfidf_predicted.append(float(svr_tfidf_predicted_score))
        knnr_extra_features_predicted.append(float(knnr_extra_features_predicted_score))
        svr_extra_features_predicted.append(float(svr_extra_features_predicted_score))
        knnr_lda_predicted.append(float(knnr_lda_predicted_score))
        svr_lda_predicted.append(float(svr_lda_predicted_score))
        knnr_tfidf_extra_predicted.append(float(knnr_tfidf_extra_predicted_score))
        svr_tfidf_extra_predicted.append(float(svr_tfidf_extra_predicted_score))
        knnr_lda_extra_predicted.append(float(knnr_lda_extra_predicted_score))
        svr_lda_extra_predicted.append(float(svr_lda_extra_predicted_score))

        print colored('essay #%d tested' % idx, 'cyan')
        index += 1

    # pickle data
    pickle.dump(actual, open('data/set%d_actual_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_predicted, open('data/set%d_knnr_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_predicted, open('data/set%d_svr_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_normalized_predicted, open('data/set%d_knnr_normalized_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_normalized_predicted, open('data/set%d_svr_normalized_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_tfidf_predicted, open('data/set%d_knnr_tfidf_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_tfidf_predicted, open('data/set%d_svr_tfidf_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_extra_features_predicted, open('data/set%d_knnr_statistics_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_extra_features_predicted, open('data/set%d_svr_statistics_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_lda_predicted, open('data/set%d_knnr_lda_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_lda_predicted, open('data/set%d_svr_lda_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_tfidf_extra_predicted, open('data/set%d_knnr_tfidf_statistics_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_tfidf_extra_predicted, open('data/set%d_svr_tfidf_statistics_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(knnr_lda_extra_predicted, open('data/set%d_knnr_lda_statistics_predicted_scores.pkl' % i, 'w+'))
    pickle.dump(svr_lda_extra_predicted, open('data/set%d_svr_lda_statistics_predicted_scores.pkl' % i, 'w+'))
    print colored('essay set%d data dumped' % i, 'grey')

    print colored('ESSAY SET %d' % i, 'green', attrs=['bold'])
    knnr_actual,knnr_predicted = filter_nan(actual, knnr_predicted)
    print colored('(RAW) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_actual, knnr_predicted), mean_absolute_error(knnr_actual, knnr_predicted)), 'green', attrs=['bold'])
    svr_actual,svr_predicted = filter_nan(actual, svr_predicted)
    print colored('(RAW) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_actual, svr_predicted), mean_absolute_error(svr_actual, svr_predicted)), 'green', attrs=['bold'])

    knnr_normalized_actual,knnr_normalized_predicted = filter_nan(actual, knnr_normalized_predicted)
    print colored('(NORMALIZED) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_normalized_actual, knnr_normalized_predicted), mean_absolute_error(knnr_normalized_actual, knnr_normalized_predicted)), 'green', attrs=['bold'])
    svr_normalized_actual,svr_normalized_predicted = filter_nan(actual, svr_normalized_predicted)
    print colored('(NORMALIZED) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_normalized_actual, svr_normalized_predicted), mean_absolute_error(svr_normalized_actual, svr_normalized_predicted)), 'green', attrs=['bold'])

    knnr_tfidf_extra_actual,knnr_tfidf_extra_predicted = filter_nan(actual, knnr_tfidf_extra_predicted)
    print colored('(TFIDF + STATISTICS) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_tfidf_extra_actual, knnr_tfidf_extra_predicted), mean_absolute_error(knnr_tfidf_extra_actual, knnr_tfidf_extra_predicted)), 'green', attrs=['bold'])
    svr_tfidf_extra_actual,svr_tfidf_extra_predicted = filter_nan(actual, svr_tfidf_extra_predicted)
    print colored('(TFIDF + STATISTICS) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_tfidf_extra_actual, svr_tfidf_extra_predicted), mean_absolute_error(svr_tfidf_extra_actual, svr_tfidf_extra_predicted)), 'green', attrs=['bold'])

    knnr_lda_extra_actual,knnr_lda_extra_predicted = filter_nan(actual, knnr_lda_extra_predicted)
    print colored('(LDA + STATISTICS) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_lda_extra_actual, knnr_lda_extra_predicted), mean_absolute_error(knnr_lda_extra_actual, knnr_lda_extra_predicted)), 'green', attrs=['bold'])
    svr_lda_extra_actual,svr_lda_extra_predicted = filter_nan(actual, svr_lda_extra_predicted)
    print colored('(LDA + STATISTICS) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_lda_extra_actual, svr_lda_extra_predicted), mean_absolute_error(svr_lda_extra_actual, svr_lda_extra_predicted)), 'green', attrs=['bold'])

    knnr_tfidf_actual,knnr_tfidf_predicted = filter_nan(actual, knnr_tfidf_predicted)
    print colored('(TFIDF) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_tfidf_actual, knnr_tfidf_predicted), mean_absolute_error(knnr_tfidf_actual, knnr_tfidf_predicted)), 'green', attrs=['bold'])
    svr_tfidf_actual,svr_tfidf_predicted = filter_nan(actual, svr_tfidf_predicted)
    print colored('(TFIDF) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_tfidf_actual, svr_tfidf_predicted), mean_absolute_error(svr_tfidf_actual, svr_tfidf_predicted)), 'green', attrs=['bold'])

    knnr_extra_features_actual,knnr_extra_features_predicted = filter_nan(actual, knnr_extra_features_predicted)
    print colored('(STATISTICS) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_extra_features_actual, knnr_extra_features_predicted), mean_absolute_error(knnr_extra_features_actual, knnr_extra_features_predicted)), 'green', attrs=['bold'])
    svr_extra_features_actual,svr_extra_features_predicted = filter_nan(actual, svr_extra_features_predicted)
    print colored('(STATISTICS) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_extra_features_actual, svr_extra_features_predicted), mean_absolute_error(svr_extra_features_actual, svr_extra_features_predicted)), 'green', attrs=['bold'])

    knnr_lda_actual,knnr_lda_predicted = filter_nan(actual, knnr_lda_predicted)
    print colored('(LDA) KNN MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(knnr_lda_actual, knnr_lda_predicted), mean_absolute_error(knnr_lda_actual, knnr_lda_predicted)), 'green', attrs=['bold'])
    svr_lda_actual,svr_lda_predicted = filter_nan(actual, svr_lda_predicted)
    print colored('(LDA) SVM MEAN SQUARE, ABSOLUTE: ', 'cyan'), colored('%f, %f' % (mean_squared_error(svr_lda_actual, svr_lda_predicted), mean_absolute_error(svr_lda_actual, svr_lda_predicted)), 'green', attrs=['bold'])

###################
#       MAIN      #
###################

def main():
    essay_sets = [1,2,3,4,5,6,7,8]
    # essay_sets = [4,5,6,7,8]
    # essay_sets = [7,8]

    for essay_set in essay_sets:
        combined_test(essay_set)

if __name__ == "__main__":
    main()      