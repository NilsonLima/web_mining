import pandas as pd

import pickle
import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from sklearn.utils import shuffle

def preprocess(tweet):
    """
        preprocess a tweet eliminating urls, mentions and other stuff
        @tweet: raw tweet text
        @ret: preprocessed tweet
        @rtype: string
    """

    #lower tweet text
    tweet = tweet.lower( )

    #remove url's
    tweet = re.sub(r'((www\.[^\s]+)|(http[s]?://[^\s]+))', ' URL ', tweet)

    #remove mentions
    tweet = re.sub(r'@[^\s]+', ' ATUSER ', tweet)

    # #replace <3 symbol with 'HEART'
    # tweet = re.sub(r'&lt;3', ' HEART ', tweet)
    #
    # #remove &, > and < symbols
    # tweet = re.sub(r'(&amp;)|(&gt;)|(&lt;)', ' ', tweet)
    #
    # #replace happy emoticons with 'SMILE'
    # tweet = re.sub(r'(?:[:=;][oO\-]?[D\)\]\]pP])', ' SMILE ', tweet)
    #
    # #replace sad emoticions with 'SAD'
    # tweet = re.sub(r'(?:[:=;][oO\-]?[(\/\]\\oO])', ' SAD ', tweet)
    #
    # #remove punctuations
    # tweet = re.sub('[%s]' % re.escape(string.punctuation), ' ', tweet)

    #remove additional white spaces
    tweet = re.sub(r'[\s]+', ' ', tweet)

    #trim tweet
    tweet = tweet.strip( )

    return tweet


def process(df, path):
    """
        preprocess all tweets from data frame
        @df: tweets data frame object
        @path: string path to save tweets and labels
        @ret: list of list of tweets and respective labels
        @rtype: void
    """

    labels = [ ]
    tweets = [ ]
    for i, row in df.iterrows( ):
        tweet = preprocess(row.text)
        tweets.append(tweet)
        labels.append(row.airline_sentiment)

    with open(path, "wb") as f:
        pickle.dump([tweets, labels], f)

    return

def main( ):
    """
        main function
        @rtype: void
    """

    df = pd.read_csv(csvpath, encoding = "UTF-8")

    df_positive = shuffle(df.loc[df.airline_sentiment == "positive"][["text", "airline_sentiment"]])
    df_neutral = shuffle(df.loc[df.airline_sentiment == "neutral"][["text", "airline_sentiment"]])
    df_negative = shuffle(df.loc[df.airline_sentiment == "negative"][["text", "airline_sentiment"]])

    positive_train = df_positive[0 : train_bound]
    neutral_train = df_neutral[0 : train_bound]
    negative_train = df_negative[0 : train_bound * nratio]

    positive_test = df_positive[train_bound : count]
    neutral_test = df_neutral[train_bound : count]
    negative_test = df_negative[train_bound * nratio : nratio * count]

    df = pd.concat([positive_train, neutral_train, negative_train])
    process(df, train_pickle)

    df = pd.concat([positive_test, neutral_test, negative_test])
    process(df, test_pickle)

    return


if __name__ == '__main__':
    count = 2360
    train = 0.8
    nratio = 2

    train_bound = int(train * count)

    csvpath = "Tweets.csv"
    train_pickle = "train.pickle"
    test_pickle = "test.pickle"

    main( )
