# -*- coding: utf-8 -*-
"""Moviereview1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BOuGgDjhXcpaqzP3jcym2KwsLqO7RZl6
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os
import warnings
import matplotlib
from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('stopwords')

import pickle

df = pd.read_csv("/IMDB Dataset.csv")

df.head()

df.shape

df.isnull().sum()

df.describe()

df.info()

df['sentiment'].unique()

df['sentiment'].value_counts()

sentiment_mapping = {'positive': 0, 'negative': 1}
df['sentiment_code'] = df['sentiment'].map(sentiment_mapping)
sns.countplot(data=df, x='sentiment_code')
plt.show()

label = LabelEncoder()
df['sentiment'] = label.fit_transform(df['sentiment'])

df.head()

X = df['review']
y = df['sentiment']

corpus

ps = PorterStemmer()
corpus = []

for i in range(len(X)):
    print(i)
    review = re.sub("[^a-zA-Z]", " ", X[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 5000)
X = cv.fit_tranform(corpus).toarray()

X.shape

X_train , X_test , Y_train , Y_test = test_train_split( X, y , test_sixe = 0 , random_state = 101)

X_train.shape , X_test.shape , Y_train.shape , Y_test.shape

mnb = MultinomialNB()
mnb.fit(X_train , Y_train)

pred = mnb.predict(X_test)

print(accuracy_score(Y_test,pred))
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))

pd.DataFrame(np.c_[Y_test , pred], columns = ['Actual', 'Predicted'])

pickle.dump(cv , open("count-vectorizer.pkl", "wb"))
pickle.dump(mnb , open("Movies_review_classification.pkl" , "wb"))

save_cv = pickle.load(open('count-Vectorizer.pkl', 'rb'))
model = pickle.load(open('Movies_review_classification.pkl' , 'rb'))

def test_model(sentence):
  sen = save_cv.transform([sentence]).toarray()
  res  = model.predict(sen)[0]
  if res ==1:
    return 'positive review'
  else:
    return ' negative review'

sen = 'tis is wonderful movie'
res = test
print(res)

sen = 'this is the worst movie '
res = test
print(res)

