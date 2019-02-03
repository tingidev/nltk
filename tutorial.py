# tutorial.py
# https://www.youtube.com/watch?v=h44hI7lr8w4&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=1

import nltk
import re
import random
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.corpus import state_union
from nltk.corpus import gutenberg
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
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

"""
NE Type Examples
ORGANIZATION    Georgia-Pacific Corp.
PERSON          President Obama
LOCATION        Mount Everest
DATE            29-03-2008
TIME            1:30 pm
MONEY           175 million Dollars
PERCENT         18.75%
FACILITY        Stonehenge
GPE             South East Asia

Regular Expression Identifiers:
\d  any number
\D  anything but a number
\s  space
\S  anything but a space
\w  any character
\W  anything but a character
.   any character, except for a newline
\b  whitespace around words
\.  a period

Regular Expression Modifiers:
{x} expect x amount or range x1-x2
+   match 1 or more
?   match 0 or 1
*   match 0 or more
$   match end of string
^   match beginning of string
|   either, or
[]  range or "variance"

Regular Expression Whitespace Characters:
\n  new line
\s  space
\t  tab
\e  escape
\f  form feed
\r  return

DONT FORGET!:
. + * ? [] $ ^ () {} | \  (escape them)

POS Tag List:
CC      Coordinating conjunction
CD      Cardinal number
DT      Determiner
E       Existential there
FW      Foreign word
IN      Preposition or subordinating conjunction
JJ      Adjective
JJR     Adjective, comparative
JJS     Adjective, superlative
LS      List item marker
MD      Modal
NN      Noun, singular or mass
NNS     Noun, plural
NNP     Proper noun, singular
NNPS    Proper noun, plural
PDT     Predeterminer
POS     Possessive ending
PRP     Personal pronoun
PRP$    Possessive pronoun
RB      Adverb
RBR     Adverb, comparative
RBS     Adverb, superlative
RP      Particle
SYM     Symbol
TO      to
UH      Interjection
VB      Verb, base form
VBD     Verb, past tense
VBG     Verb, gerund or present participle
VBN     Verb, past participle
VBP     Verb, non-3rd person singular present
VBZ     Verb, 3rd person singular present
WDT     Wh-determiner
WP      Wh-pronoun
WP$     Possessive wh-pronoun
WRB     Wh-adverb

# Tokeninzing by sentence vs. keywords
# Corpora - body of text (e.g. medical journal, English language)
# Lexicon - words and their means (e.g. dictionary)

# Part 1 - tokenize
example_text = "Hello Mr. Smith, how are you doing today? The weather is sunny and python is awesome. The sky is very blue. You should not eat cardboard."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))
for w in word_tokenize(example_text):
    print(w)

# Part 2 - stop words
example_sentence = "This is an example sentence showing off stop word filtration."
stop_words = set(stopwords.words("english"))
print(stop_words)
words = word_tokenize(example_sentence)
filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)

# Part 3 - stemming
example_words = ["python","pythoner","pythoning","pythoned","pythonized"]
ps = PorterStemmer()
for w in example_words:
    print(ps.stem(w))
new_text = "It is very important to python properly when pythoning with python. All pythoners were pythonized at least once."
words = word_tokenize(new_text)
stemmed_words = [ps.stem(w) for w in words]
print(stemmed_words)

# Part 4, 5, 6 & 7 - speech tagging, chunking, chinking & named entity recognition
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text )
def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=True)
            print(namedEnt)

            # chunkGram = r"Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"
            # chunkGram = r"Chunk: {<.*>+}
            #                       }<VB.?|IN|DT|TO>+{"
            # chunkParser = nltk.RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            # print(chunked)
            # chunked.draw()
            #print(tagged)
    except Exception as e:
        print(str(e))
process_content()

# Part 5 - regular expressions
exampleString = """
#Jessica is 15 years old, and Daniel is 27 years old.
#Edward is 97, and his grandfather, Oscar, is 102.
"""
ages = re.findall(r'\d{1,3}',exampleString)
names = re.findall(r'[A-Z][a-z]*',exampleString)
print(ages)
print(names)
ageDict={}
x=0
for n in names:
    ageDict[n] = ages[x]
    x+=1
print(ageDict)

# Part 8 - lemmatizing
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better")) #default pos = noun
print(lemmatizer.lemmatize("better",pos="a")) #set to verb
print(lemmatizer.lemmatize("worse",pos="a")) #set to verb

# Part 9 - nltk corpora
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])

# Part 10 - WordNet
syns = wordnet.synsets("program")
print(syns) # list of synonyms
print(syns[0].lemmas()[0].name()) # word
print(syns[0].definition()) # definition
print(syns[0].examples()) # example
synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            for a in l.antonyms():
                antonyms.append(a.name())
print(set(synonyms))
print(set(antonyms))
w1 = wordnet.synsets("ship")[0]
w2 = wordnet.synsets("boat")[0]
w3 = wordnet.synsets("car")[0]
w4 = wordnet.synsets("cat")[0]
print("set2:",w1.wup_similarity(w2))
print("set3:",w1.wup_similarity(w3))
print("set4:",w1.wup_similarity(w4))
"""
# Part 11, 12, 13 & 14,15,16 & 17 - text classification, words as features, naiv bayes, pickle, scikit-Learn, voting algorhytms and bias
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]
# random.shuffle(documents)
# print(documents[0])
# all_words = []
# for w in movie_reviews.words():
#     all_words.append(w.lower())

# Part 19 - Allowed word types
# [J]=adjective, [R]=adverb, [V]=verb
allowed_word_types=["J"]

# Part 18 - New training data
short_pos = open("C:/Users/jvw/Documents/Python Scripts/nltk/data/positive.txt","r").read()
short_neg = open("C:/Users/jvw/Documents/Python Scripts/nltk/data/negative.txt","r").read()
all_words = []
documents = []

for r in short_pos.split('\n'):
    documents.append( (r,"pos") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append( (r,"neg") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# save_documents = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/documents.pickle","wb")
# pickle.dump(documents, save_documents)
# save_documents.close()

# for r in short_pos.split('\n'):
#     documents.append( (r,"pos") )
# for r in short_neg.split('\n'):
#     documents.append( (r,"neg") )
# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)
# for w in short_pos_words:
#     all_words.append(w.lower())
# for w in short_neg_words:
#     all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["stupid"])

word_features = [w[0] for w in all_words.most_common(5000)]
save_word_features = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    # words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_featuresets = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

#Short reviews
random.shuffle(featuresets)
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
#Positive data
# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]
#Negative data
# training_set = featuresets[100:]
# testing_set = featuresets[:100]
# posterior = prior occurences * likelihood / evidence

#NAIV BAYES
classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features(15)
print("Original Naive Bayes accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_original_naiv_bayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
# classifier_f = open("naivebayes.pickle","rb")
# classifier_loaded = pickle.load(classifier_f)
# classifier_f.close()

#MNB
MNB_classifier = SklearnClassifier(MultinomialNB()) # Multinomial
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_MNB.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

# GNB_classifier = SklearnClassifier(GaussianNB()) # Gaussian (not working)
# GNB_classifier.train(training_set)
# print("GNB_classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

#BNB
BNB_classifier = SklearnClassifier(BernoulliNB()) # Bernoulli
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_BNB.pickle","wb")
# pickle.dump(BNB_classifier, save_classifier)
# save_classifier.close()

#LOGISTIC REGRESSION
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_logistic_regression.pickle","wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()

#SGD
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_SGD.pickle","wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

#SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_SVC.pickle","wb")
# pickle.dump(SVC_classifier, save_classifier)
# save_classifier.close()

#LINEAR SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_linear_SVC.pickle","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

#NUSVC
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
#
# save_classifier = open("C:/Users/jvw/Documents/Python Scripts/nltk/pickled_algorithms/classifier_MNB.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  SVC_classifier,
                                  LinearSVC_classifier)

print("Voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence:", voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence:", voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence:", voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[3][0]),   "Confidence:", voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence:", voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)
