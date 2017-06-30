import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def tokenize(tweet):
    """
        tokenize preprocessed tweet and remove stopwords
        @tweet: preprocessed tweet
        @ret: tokenized tweet
        @rtype: string list
    """

    tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)
    words = stopwords.words("english") + ["ATUSER", "URL", "RT", "via"]

    tokenized =  tokenizer.tokenize(tweet)
    tokenized = [t for t in tokenized if t not in words]
    #tokenized = [t for t in tokenized if t.isalpha( ) == True]

    return tokenized

def airline(name, classifier = None):
    """
        airline sentiment occurences by date
        @name: airline name
        @classifier: classifier that predicts sentiment
        @ret: dates and sentiment occurences
        @rtype: tuple
    """

    df = pd.read_csv(tweetspath, encoding = "UTF-8")
    airline = df.loc[df.airline == name][["tweet_created", "text", "airline_sentiment"]]

    dates = airline.tweet_created.tolist( )
    dates = set([d.split( )[0] for d in dates])
    dates  = sorted(dates, key = lambda d: datetime.strptime(d, '%Y-%m-%d'))

    sentiment = airline.airline_sentiment.tolist( )
    sentiment = set(sentiment)

    sentiments = [ ]

    if classifier == None :
        for s in sentiment:
            sent = [ ]
            for d in dates:
                dated = airline.loc[airline.tweet_created.str.contains(d) == True]["airline_sentiment"].value_counts( )
                sent.append(dated.loc[s])

            sentiments.append(sent)

    else:
        for s in sentiment:
            sent = [ ]
            for d in dates:
                x_pred = airline.loc[airline.tweet_created.str.contains(d) == True]["text"].tolist( )
                pred = list(classifier.predict(x_pred))

                sent.append(pred.count(s))

            sentiments.append(sent)

    return dates, sentiments

def plot(dates, y):
    """
        plot stacked area
        @x: dates
        @y: sentiment occurences
        @rtype: void
    """

    x = [datetime.strptime(d, "%Y-%m-%d").date( ) for d in dates]
    xfmt = mdates.DateFormatter("%a %d")

    fig, ax = plt.subplots( )
    ax.stackplot(x, y, colors = ["#9ecae1", "#6baed6", "#3182bd"])
    ax.xaxis.set_major_formatter(xfmt)

    plt.xticks(rotation = -30)
    plt.show( )

    return

def main( ):
    """
        main function
        @rtype: void
    """

    with open(clf_pickle, "rb") as f:
        clf = pickle.load(f)

    x, y = airline("Virgin America", clf)
    plot(x, y)

    return

if __name__ == '__main__':
    tweetspath = "Tweets.csv"
    clf_pickle = "clf.pickle"

    main( )
