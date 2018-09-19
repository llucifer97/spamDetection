#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 23:08:00 2018

@author: ayush
"""


import nltk

#nltk.download_shell()

messages = [line.rstrip() for line in open('SMSSpamCollection')]

print(len(messages))

messages[50]

for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep = '\t', names = ['label','message'])

messages.head()

messages.describe()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)


messages.head()

import matplotlib.pyplot as plt
import seaborn as sns


messages['length'].plot.hist(bins = 150)
messages['length'].describe()

messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column = 'length',by = 'label',bins = 60,figsize = (12,4))

##############

import string

mess = 'sample message:notice it ha spuncuations.'

nopunc = [c for c in mess if c not in string.punctuation] 
nopunc

from nltk.corpus import stopwords

#stopwords.words('english')

nopunc = ''.join(nopunc)

nopunc.split()

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess

def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """

    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


messages.head()

messages['message'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]

print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4.shape)

bow_transformer.get_feature_names()[4068]

####################


message_bow = bow_transformer.transform(messages['message'])
print('shape of sparse matrix:',message_bow.shape)

message_bow.nnz


sparsity = (100.0 * message_bow.nnz/(message_bow.shape[0] * message_bow.shape[1]))
('sparsity: {}'.format(sparsity))


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(message_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

message_tfidf = tfidf_transformer.transform(message_bow)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(message_tfidf,messages['label'])

spam_detect_model.predict(tfidf4)[0]

messages['label'][3]

all_pred = spam_detect_model.predict(message_tfidf)

all_pred

from sklearn.cross_validation import train_test_split

msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size = 0.3)


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
          ('bow',CountVectorizer(analyzer = text_process)),
          ('tfidf',TfidfTransformer()),
          ('classifier',MultinomialNB())
          ])

    
pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report


print(classification_report(label_test,predictions))













































