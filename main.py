from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

"""data sets"""
train_courts = pd.read_csv("data.csv", delimiter=',', header=None, names=['text', 'court']).sample(frac=1)
test_courts = pd.read_csv("data_test.csv", delimiter=',', header=None, names=['text', 'court'])

"""categories"""
vec = CountVectorizer()
train_vec = vec.fit_transform(train_courts["text"])
test_vec = vec.transform(test_courts["text"])

tdidf = TfidfTransformer()
train_tfidf = tdidf.fit_transform(train_vec)
test_tfidf = tdidf.fit_transform(test_vec)

clf = RandomForestClassifier()
clf.fit(train_tfidf, train_courts["court"].values)
predict = clf.predict(test_tfidf)

score = clf.score(test_tfidf, test_courts["court"].values)
print(int(score*100), "%", " predicted")
print(predict)
print(test_courts["court"].values)

