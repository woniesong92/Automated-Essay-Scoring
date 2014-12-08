import feature_extractor
from gensim.models.doc2vec import Doc2Vec
import parser
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pdb


model = Doc2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary = True)
print "MODEL LOADED"

f = open('stopwords.txt')
stoplist = set(line.split('\n')[0] for line in f)

def filter_essay(essay):
    stop_removed = filter(lambda x: x not in stoplist, essay.split())
    all_filtered = filter(lambda x: x in model.vocab, stop_removed)
    return all_filtered

def filter_essays(essays):
    return [filter_essay(essay) for essay in essays]



def calc_similarity(i1, i2):
    return model.n_similarity(i1, i2)

def classify(k, instance, training_data, training_scores):
    similarity = np.array([calc_similarity(instance, x) for x in training_data])
    neighbors_indices = np.argpartition(-similarity, k)[:k]
    predicted_score = 0
    total_weights = 0
    for neighbor in neighbors_indices:
        total_weights += float(similarity[neighbor])
        predicted_score += float(similarity[neighbor]) * float(training_scores[neighbor])
    print predicted_score
    print total_weights
    return float(predicted_score)/total_weights



myParser = parser.Parser()
myFeatureExtractor = feature_extractor.FeatureExtractor()

i=1

training_examples = myParser.parse("data/set%d_train.tsv" % i)
test_examples = myParser.parse("data/set%d_test.tsv" % i)



train_essays, train_scores = myFeatureExtractor.extract_essay_and_scores(training_examples)
train_essays = filter_essays(train_essays)


test_essays, test_scores = myFeatureExtractor.extract_essay_and_scores(test_examples)
test_essays = filter_essays(test_essays)




predicted_scores = []
actual_scores = []

k = 5

print "starting classifying"
for i in range(len(test_essays)):
    actual = test_scores[i]
    predicted = classify(k, test_essays[i], train_essays, train_scores)
    predicted_scores.append(predicted)
    actual_scores.append(actual)
    print "finished iteration %d" % i 

pdb.set_trace()











