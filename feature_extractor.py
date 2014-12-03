import parser
from tfidf_vectorizer import Tfidf_Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import pdb
import numpy as np
import pickle
import nltk
import threading
import math


tag_dict = {'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,
            'LS':9,'MD':10,'NN':11,'NNS':12,'NNP':13,'NNPS':14,'PDT':15,
            'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,
            'SYM':23,'TO':24,'UH':25,'VB':26,'VBD':27,'VBG':28,'VBN':29,
            'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34,'WRB':35,'.':36,',':37,
            ':':38,'``':39,"''":40,'#':41,'$':42,'-NONE-':43}


# Given that we already have "essays.pkl"
# extract tfidf, word_count, number of different words, words average length
# More ideas: complexity of the parse trees of the essay?
class FeatureExtractor:
    def extract_essay_and_scores(self, train_examples):
        essays= []
        scores = []
        for train_example in train_examples:
            essay = self.remove_symbols(train_example["essay"])
            essays.append(essay)
            scores.append(train_example["domain1_score"])
        return (essays, scores)

    # List of features for each essay
    # 1. tfidf
    # 2. word count
    # 3. number of different words
    # 4. words' average length 
    # 5. Part of speech tagging
    # TODO: parse trees' complexity?
    def transform_training_examples(self, train_examples):
        # pdb.set_trace()
        essays, scores = self.extract_essay_and_scores(train_examples)
        # 1. tfidf 
        tfidf_vectorizer = Tfidf_Vectorizer()
        (tfidf, tfs) = tfidf_vectorizer.get_tfidf_vectors(essays)
        matrix = np.array(tfs.todense())
        print "DONE TFIDF"




        # EXTRA FEATURES
        # 2. word count
        word_counts_features = [len(essay.split()) for essay in essays]

        print "DONE WORD COUNT"
        # 3. number of different words
        num_diff_words_features = [self.num_diff_words(essay) for essay in essays]

        print "DONE NUM DIFF WORDS"
        # 4. average length of words
        words_avg_length_features = [self.words_avg_length(essay) for essay in essays]

        print "AVERAGE LENGTH OF WORDS"

        extra_features = []
        for essay in essays:
            vector = [len(essay.split()), self.num_diff_words(essay), self.words_avg_length(essay)]
            extra_features.append(self.l2_norm(vector))



        matrix = np.concatenate((extra_features, matrix), 1)

        # 5. part of speech tagging
        # pos_distributions = [self.part_of_speech_tagging(essay) for essay in essays]
        # print "TAGGING DONE"
        # matrix = np.concatenate((pos_distributions, matrix),1)

        # print "UPDATED TAGGING WITH MATRIX"

        # Add the scores 
        # matrix = self.update_matrix(matrix, scores)


        return matrix, tfidf

    def transform_test_examples(self, test_examples, tfidf):
        essays, scores = self.extract_essay_and_scores(test_examples)

        # 1. tfidf
        matrix = np.array(tfidf.transform(essays).todense())
        print "DONE TFIDF"

        # EXTRA FEATURES
        # 2. word count
        word_counts_features = [len(essay.split()) for essay in essays]

        print "DONE WORD COUNT"
        # 3. number of different words
        num_diff_words_features = [self.num_diff_words(essay) for essay in essays]

        print "DONE NUM DIFF WORDS"
        # 4. average length of words
        words_avg_length_features = [self.words_avg_length(essay) for essay in essays]

        print "AVERAGE LENGTH OF WORDS"

        extra_features = []
        for essay in essays:
            vector = [len(essay.split()), self.num_diff_words(essay), self.words_avg_length(essay)]
            extra_features.append(self.l1_norm(vector))



        matrix = np.concatenate((extra_features, matrix), 1)

        # 5. part of speech tagging
        # pos_distributions = [self.part_of_speech_tagging(essay) for essay in essays]
        # print "TAGGING DONE"
        # matrix = np.concatenate((pos_distributions, matrix),1)

        # print "UPDATED TAGGING WITH MATRIX"

        # Add the scores 
        # matrix = self.update_matrix(matrix, scores)

        return matrix



    ####### HELPER METHODS ######
    def update_matrix(self, matrix, new_features):
        return np.insert(matrix, 0, new_features, 1)

    def remove_symbols(self, essay):
        # return re.sub(r'[^\w+.+\`+\'+#+!+,+$+:]', ' ', essay)
        return re.sub(r'[^\w]', ' ', essay.lower())

    def words_avg_length(self, essay):
        words = essay.split(" ")
        lengths = [len(w) for w in words]
        avg_word_length = sum(lengths) / len(lengths)
        return avg_word_length

    def num_diff_words(self, essay):
        unique_words = set()
        words = essay.split(" ")
        for word in words:
            if word not in unique_words:
                unique_words.add(word)
        return len(unique_words)

    def part_of_speech_tagging(self, essay):
        text = nltk.word_tokenize(essay)
        tagged = nltk.pos_tag(text)
        count = [0]*len(tag_dict)
        for token in tagged:
            count[tag_dict[token[1]]] += 1
        total = sum(count)
        count = [float(pos)/total for pos in count]
        return np.array(count)

    def l1_norm(self, vector):
        s = sum(vector)
        return [value/float(s) for value in vector]

    def l2_norm(self, vector):
        mag = math.sqrt(sum([value * value for value in vector]))
        return [value/float(mag) for value in vector]

class Worker(threading.Thread):
    def __init__(self, i):
        threading.Thread.__init__(self)
        self.i=i

    def run(self):
        i = self.i
        myParser = parser.Parser()
        myFeatureExtractor = FeatureExtractor()
        training_examples = myParser.parse("data/set%d_train.tsv" % i)
        test_examples = myParser.parse("data/set%d_test.tsv" % i)

        train_matrix, tfidf = myFeatureExtractor.transform_training_examples(training_examples)
        test_matrix = myFeatureExtractor.transform_test_examples(test_examples, tfidf)
        
        train_pkl = open("data/set_%d_train_matrix.pkl" % i, "w+")
        test_pkl = open("data/set_%d_test_matrix.pkl" % i, "w+")

        pickle.dump(train_matrix, train_pkl)
        pickle.dump(test_matrix, test_pkl)

        train_pkl.close()
        test_pkl.close()





if __name__ == "__main__":
    myParser = parser.Parser()
    myFeatureExtractor = FeatureExtractor()
    for i in range(1,9):
        # worker = Worker(i)
        # worker.start() 
        # SEQUENTIAL
        training_examples = myParser.parse("data/set%d_train.tsv" % i)
        test_examples = myParser.parse("data/set%d_test.tsv" % i)

        train_matrix, tfidf = myFeatureExtractor.transform_training_examples(training_examples)
        test_matrix = myFeatureExtractor.transform_test_examples(test_examples, tfidf)
        
        train_pkl = open("data/set_%d_train_matrix.pkl" % i, "w+")
        test_pkl = open("data/set_%d_test_matrix.pkl" % i, "w+")

        pickle.dump(train_matrix, train_pkl)
        pickle.dump(test_matrix, test_pkl)

        train_pkl.close()
        test_pkl.close()





    # training_examples = myParser.parse("data/training_set_rel3.tsv")
    # #training_examples = myParser.parse("data/set1_train.tsv")

    # #training_examples_filtered = list(filter(lambda x: x["essay_set"]== "1", training_examples))

    # training_dict = {}
    # for example in training_examples:
    #     key = example["essay_id"]
    #     value = example
    #     training_dict[key] = value

    
    


    # test_examples = myParser.parse("data/test_set.tsv")
    # test_examples = list(filter(lambda x: x["essay_set"]== "1", test_examples))

    # pdb.set_trace()

    # test_examples_new = []
    # for example in test_examples:
    #     score = training_dict[example["domain1_predictionid"]]["domain1_score"]
    #     example["domain1_score"] = score
    #     test_examples_new.append(example)




    # test_matrix = myFeatureExtractor.transform_training_examples(test_examples_new)
    # test_pkl = open("test_essay_set_1.pkl", "w+")
    # pickle.dump(test_matrix, test_pkl)




