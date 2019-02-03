# sentiment_mod.py

import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def find_features(document):
    words = word_tokenize(document)
    # words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Documents
documents_f = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

# Word features
word_features_f = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/word_features.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

# Feature sets
featuresets_f = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

# Data
random.shuffle(featuresets)
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# Classifiers
open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_original_naiv_bayes.pickle","rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_MNB.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_BNB.pickle","rb")
BNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_logistic_regression.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_SGD.pickle","rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

# open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_SVC.pickle","rb")
# SVC_classifier = pickle.load(open_file)
# open_file.close()
#
# open_file = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_linear_SVC.pickle","rb")
# LinearSVC_classifier = pickle.load(open_file)
# open_file.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
