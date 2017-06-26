
# coding: utf-8

# In[1]:

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
import pandas as pd
import re


# In[2]:

df = pd.read_csv('Tweets.csv')


# In[3]:

col_orig = list(df.columns)
col_new = ['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'text']


# In[4]:

"""
    Testing area
"""


df = df[col_new]
df_text = df[['airline_sentiment', 'text']]
# test_text = df[['airline_sentiment', 'text']][:500]


# In[5]:

def preprocess_tweet(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    """
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    """
    #Remove www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #Replace @username with username
    tweet = re.sub(r'@([^\s]+)', r'\1', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet


# In[6]:

"""
    Testing area
"""


tweets_data = []
for text in df_text['text']:
# for text in test_text['text']:
    preproc_text = preprocess_tweet(text)
    tokenized_text = nltk.word_tokenize(preproc_text.decode('utf-8'))
    for i, token in enumerate(tokenized_text):
        tokenized_text[i] = token.encode('utf-8')
    tweets_data.append(tokenized_text)


# In[7]:

"""
    Testing area
"""


for i, row in enumerate(tweets_data):
    tweets_data[i] = (row, df_text['airline_sentiment'][i])
#     tweets_data[i] = (row, test_text['airline_sentiment'][i])


# In[8]:

"""
    Testing area
"""

training_docs = tweets_data[:10000]
testing_docs = tweets_data[10000:]


# In[9]:

sentim_analyzer = SentimentAnalyzer()


# In[10]:

all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)


# In[11]:

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)


# In[12]:

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)


# In[13]:

for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

