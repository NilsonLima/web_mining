import re
import pickle
import argparse

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

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

    #remove additional white spaces
    tweet = re.sub(r'[\s]+', ' ', tweet)

    #trim tweet
    tweet = tweet.strip( )

    return tweet

def tokenize(tweet):
    """
        tokenize preprocessed tweet and remove stopwords
        @tweet: preprocessed tweet
        @ret: tokenized tweet
        @rtype: list of strings
    """

    tokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)
    words = stopwords.words("english") + ["ATUSER", "URL", "RT", "via"]

    tokenized =  tokenizer.tokenize(tweet)
    tokenized = [t for t in tokenized if t not in words]
    #tokenized = [t for t in tokenized if t.isalpha( ) == True]

    return tokenized

def main( ):
    """
        main function
        @rtype: void
    """
    global clf_pickle

    if FLAGS.clf == "sklearn":
        clf_pickle = "clf.pickle"

    with open(clf_pickle, "rb") as f:
        clf = pickle.load(f)

    print("\nClassifier --> %s" % FLAGS.clf)
    print("Tweet --> '" + FLAGS.tweet + "'")

    if FLAGS.clf == "nltk":
        tweet = tokenize(preprocess(FLAGS.tweet))

        print("Sentiment -->  " + clf.classify(tweet) + "\n")

    elif FLAGS.clf == "sklearn":
        print("Sentiment -->  " + clf.predict([preprocess(FLAGS.tweet)])[0] + "\n")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser( )
    parser.add_argument('--tweet', type = str, default = "", help = "Type a tweet example abording US Airlines.")
    parser.add_argument('--clf', type = str, default = "nltk", help = "Choose which classifier to use.")

    FLAGS, unparsed = parser.parse_known_args( )

    clf_pickle = "sentiment.pickle"

    main( )
