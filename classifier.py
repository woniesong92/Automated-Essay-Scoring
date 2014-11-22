import pickle
import pdb
import random

from sklearn.svm import SVC as classifier
import numpy as np

train_pkl = open("train_essay_set_1.pkl")
# test_pkl = open("test_essay_set_1.pkl")
train_matrix = pickle.load(train_pkl)

np.random.shuffle(train_matrix)
train = train_matrix[:1000, :]
test = train_matrix[1000:, :]



x_train = train[:,1:]
y_train = train[:,0]

x_test = test[:,1:]
y_test = test[:,0]
y_test = y_test.reshape( -1, 1 )



clf = classifier(5)
clf = clf.fit(x_train, y_train.ravel())


num_correct = 0
errors = {}
for i in range(len(test)):
    feature = x_test[i]
    predicted = clf.predict(feature)
    diff = abs(predicted.tolist()[0] - y_test[i].tolist()[0])
    if diff not in errors.keys():
        errors[diff] = 0
    errors[diff] += 1
    print "predicted: " + `predicted` + " real: " + `y_test[i]`
    if predicted[0] - y_test[i][0]:
        num_correct += 1

print `errors`
print `float(num_correct)/len(test)`