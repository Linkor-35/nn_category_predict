# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

"""data sets"""
train_courts = pd.read_csv("data.csv", delimiter=',', header=None, names=['text', 'court'])
test_courts = pd.read_csv("data_test.csv", delimiter=',', header=None, names=['text', 'court'])

"""categories"""
court_categories = set(train_courts['court'].values)


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories , shuffle=True, random_state=42)

docs_new = ['God is love', 'OpenGL on the GPU is fast']


for t in twenty_train.target[:10]:
    print(twenty_train.target[t],"==>", twenty_train.data[t][:30])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))



print()
