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
from sklearn.svm import SVC as svm
from sklearn.svm import SVR as svr
import numpy as np



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



# Should be called once in the beginning
# Generate train.tsv, test.tsv, dictionary(gensim id2word), mm(gensim corpus) for each essay set
# Files saved in data/
def setUp():
    essays = {}

    # essays to dictionary
    with open('data/training_set_rel3.tsv', "r") as f:
        for idx,essay in enumerate(f):
            if idx == 0: continue
            essay_set = essay.split('\t')[1]
            try:
                essays[essay_set].append(essay)
            except KeyError:
                essays[essay_set] = [essay]

    for i in xrange(1,9):
        total = len(essays[str(i)])
        with open('data/set%d_train.tsv' % i, "w") as f1:
            for x in xrange(int(total * 0.8)):
                f1.write(essays[str(i)][x])

        with open('data/set%d_test.tsv' % i, "w") as f2:
            for x in xrange(int(total * 0.8), len(essays[str(i)])):
                f2.write(essays[str(i)][x])

    for x in xrange(1,9): 
        essay_set = x

        texts,scores = create_texts(essay_set)
        myDict = gensim.corpora.Dictionary(texts)
        myDict.save('data/set%d.dict' % essay_set)

        with open('data/set%d.scores' % essay_set, "w") as f:
            for score in scores:
                f.write(score + '\n')

        # id2word = gensim.corpora.Dictionary.load('data/set%d.dict' % essay_set)
        
        corpus = [myDict.doc2bow(text) for text in texts]

        corpus_old = corpus

        # feature selection
        bad_good_features = get_best_features(corpus, scores, 'all') #feature selected corpus
        bad_ids = []
        for i in range(len(bad_good_features)):
            if not bad_good_features[i]:
                bad_ids.append(bad_good_features[i])
        myDict.filter_tokens(bad_ids=bad_ids)

        corpus = [myDict.doc2bow(text) for text in texts]

        gensim.corpora.MmCorpus.serialize('data/set%d.mm' % essay_set, corpus)
        print colored('Essay set%d done' % essay_set, 'green', attrs=['bold'])

############################
#       HELPER METHODS     #
############################

def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

def remove_symbols(essay):
    return re.sub(r'[^\w]', ' ', essay.lower())

# Returns tokenized essay, scores
def create_texts(essay_set):
    myParser = parser.Parser()
    myFeatureExtractor = feature_extractor.FeatureExtractor()
    training_examples = myParser.parse('data/set%d_train.tsv' % essay_set)
    essays, scores = myFeatureExtractor.extract_essay_and_scores(training_examples) #list of raw essays and their scores
    documents = [remove_symbols(training_example['essay']) for training_example in training_examples]

    # remove common words
    f = open('stopwords.txt')
    stoplist = set(line.split('\n')[0] for line in f)
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    print colored('common words removed', 'cyan')

    # stemming
    texts = [stem_tokens(text, PorterStemmer()) for text in texts]


    """
    might be useful?
    """
    # remove words that appear only once
    # all_tokens = sum(texts, [])
    # tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    # texts = [[word for word in text if word not in tokens_once] for text in texts]
    # print colored('removed words that appear only once', 'cyan')

    return texts, scores

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


def get_train_features(i):
    features = pickle.load(open("data/set_%d_train_matrix.pkl" % i, 'r'))
    return features

def get_test_features(i):
    features = pickle.load(open("data/set_%d_test_matrix.pkl" % i, 'r'))
    return features

# return normalized vector
def normalize(vector):
    mag = math.sqrt(sum([value * value for value in vector]))
    return [value/float(mag) for value in vector]


###################
#       MAIN      #
###################

# Dictionaries and corpus are already saved 
def main():
    # setUp()

    myParser = parser.Parser()
    myFeatureExtractor = feature_extractor.FeatureExtractor()
    num_topics_for_lda = 30
    
    for x in xrange(1,9): 
        essay_set = x

        # prepare dictionary and corpus
        myDict = gensim.corpora.Dictionary.load('data/set%d.dict' % essay_set)
        corpus = gensim.corpora.MmCorpus('data/set%d.mm' % essay_set)

        lda = gensim.models.LdaModel(corpus=corpus, id2word=myDict, num_topics=num_topics_for_lda)

        topic_vectors = []
        for doc in corpus:
            topic_distribution = lda[doc]
            topic_vector = topic_distribution_to_vector(topic_distribution, num_topics_for_lda)
            topic_vectors.append(topic_vector)
        print colored('PREPARED TOPIC VECTORS', 'cyan')

        neigh = knn(n_neighbors=5)
        svm_classifier = svm()
        svr_classifier = svr()
        scores = []
        with open('data/set%d.scores' % essay_set) as f:
            for score in f:
                scores.append(int(score.split('\n')[0]))

        train_features_old = get_train_features(essay_set)
        train_features = np.concatenate((topic_vectors, train_features_old), 1)


        neigh.fit(train_features, scores)
        svm_classifier.fit(train_features, scores)
        # svr_classifier.fit(train_features, scores)
        print colored('PREPARED KNN, NOW LOAD TEST SET', 'cyan')
        pdb.set_trace()

        test_examples = myParser.parse("data/set%d_test.tsv" % essay_set)
        test_essays, test_scores = myFeatureExtractor.extract_essay_and_scores(test_examples) #list of raw essays and their scores
        new_test_essays = [tokenize_no_stop_words(essay) for essay in test_essays]

        #myDict.add_documents(new_test_essays)

        num_correct_neigh = 0
        num_correct_svm = 0
        num_correct_svr = 0

        index = 0
        test_features = get_test_features(essay_set)
        final_test_features = []

        for idx, test_essay in enumerate(new_test_essays):
            doc_bow = myDict.doc2bow(test_essay)
            doc_lda = lda[doc_bow]
            
            vectorized_lda = topic_distribution_to_vector(doc_lda, num_topics_for_lda)
            test_feature = np.concatenate((vectorized_lda, test_features[index]), 1)
            final_test_features.append(final_test_features)

            # pdb.set_trace()
            predicted_score = neigh.predict(test_feature)
            # print "Actual:", test_scores[idx], "Predicted:", predicted_score
            if int(test_scores[idx]) == int(predicted_score[0]):
                num_correct_neigh += 1

            predicted_score2 = svm_classifier.predict(test_feature)
            if int(test_scores[idx]) == int(predicted_score2[0]):
                num_correct_svm += 1

            index += 1

        print colored('DONE', 'green', attrs=['bold'])

        accuracy_knn = num_correct_neigh / float(len(test_scores))
        accuracy_svm = num_correct_svm / float(len(test_scores))




        print colored('ESSAY SET %d KNN ACCURACY: %f' % (essay_set, accuracy_knn), 'green', attrs=['bold'])

        print colored('ESSAY SET %d SVM ACCURACY: %f' % (essay_set, accuracy_svm), 'green', attrs=['bold'])
        pdb.set_trace()

if __name__ == "__main__":
    main()      