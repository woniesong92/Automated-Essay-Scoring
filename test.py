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



###################
#       MAIN      #
###################

def main():
    # setUp()

    myParser = parser.Parser()
    myFeatureExtractor = feature_extractor.FeatureExtractor()
    num_topics_for_lda = 30
    
    # Iterate over 8 essay sets
    for essay_set in xrange(1,9): 

        ####################
        #       LDA        #
        ####################
        # prepare dictionary and corpus
        myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % essay_set)
        corpus = gensim.corpora.MmCorpus('data/set%d.mm' % essay_set)

        lda = gensim.models.LdaModel(corpus=corpus, id2word=myDict, num_topics=num_topics_for_lda, iterations=1000, passes=3)

        topic_vectors = []
        for doc in corpus:
            topic_distribution = lda[doc]
            topic_vector = topic_distribution_to_vector(topic_distribution, num_topics_for_lda)
            topic_vectors.append(topic_vector) # normalize
        print colored('PREPARED TOPIC VECTORS', 'cyan')

        neigh = knn(n_neighbors=5, weights = 'distance')
        svm_classifier = svm()
        svr_classifier = svr()
        knnr_classifier = knnR(n_neighbors=5, weights = 'distance')

        scores = []
        with open('data/set%d.scores' % essay_set) as f:
            for score in f:
                scores.append(int(score.split('\n')[0]))

        #######################################
        #       TFIDF + other features        #
        #######################################
        train_features_old = get_train_features(essay_set)
        train_features = train_features_old
        # train_features = np.concatenate((topic_vectors, train_features_old), 1)
        print colored('PREPARED TFIDF + OTHER FEATURES VECTORS', 'cyan')

        ###############################
        #       CLASSIFICATION        #
        ###############################
        neigh.fit(train_features, scores)
        svm_classifier.fit(train_features, scores)
        knnr_classifier.fit(train_features, scores)
        print colored('PREPARED KNN, NOW LOAD TEST SET', 'cyan')


        test_examples = myParser.parse("data/set%d_test.tsv" % essay_set)
        test_essays, test_scores = myFeatureExtractor.extract_essay_and_scores(test_examples) #list of raw essays and their scores
        new_test_essays = [tokenize_no_stop_words(essay) for essay in test_essays]

        #myDict.add_documents(new_test_essays)

        num_correct_neigh = 0
        num_correct_svm = 0
        num_correct_svr = 0

        index = 0
        test_features = get_test_features(essay_set)
        
        predicted = []
        actual = []

        for idx, test_essay in enumerate(new_test_essays):
            doc_bow = myDict.doc2bow(test_essay)
            doc_lda = lda[doc_bow]
            
            vectorized_lda = topic_distribution_to_vector(doc_lda, num_topics_for_lda)
            # test_feature = np.concatenate((vectorized_lda, test_features[index]), 1)
            test_feature = test_features[index]
            

            # pdb.set_trace()
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


        print colored('DONE', 'green', attrs=['bold'])

        accuracy_knn = num_correct_neigh / float(len(test_scores))
        accuracy_svm = num_correct_svm / float(len(test_scores))




        print colored('ESSAY SET %d KNN ACCURACY: %f' % (essay_set, 100*accuracy_knn), 'green', attrs=['bold'])

        print colored('ESSAY SET %d SVM ACCURACY: %f' % (essay_set, 100*accuracy_svm), 'green', attrs=['bold'])
        print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (essay_set, mean_squared_error(actual, predicted)), 'green', attrs=['bold'])
        print colored('ESSAY SET %d KNN MEAN SQUARE: %f' % (essay_set, mean_absolute_error(actual, predicted)), 'green', attrs=['bold'])


        pdb.set_trace()

if __name__ == "__main__":
    main()      