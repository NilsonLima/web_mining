import pandas as pd
import numpy as np

import pickle
import nltk

from nltk.corpus import stopwords
from nltk.sentiment.util import extract_unigram_feats
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer

def main( ):
    """
        main function
    """

    train_data = [ ]
    with open(train_pickle, "rb") as f:
        train_data = pickle.load(f)

    test_data = [ ]
    with open(test_pickle, "rb") as f:
        test_data = pickle.load(f)

    sentiment = SentimentAnalyzer( )

    words = sentiment.all_words(train_data)
    unigrams = sentiment.unigram_word_feats(words, min_freq = 4)
    sentiment.add_feat_extractor(extract_unigram_feats, unigrams = unigrams)

    training_set = sentiment.apply_features(train_data)
    test_set = sentiment.apply_features(test_data)

    trainer = nltk.classify.NaiveBayesClassifier.train

    sentiment.train(trainer, training_set)
    sentiment.evaluate(test_set, verbose = True)

    with open(sent_pickle, "wb") as f:
        pickle.dump(sentiment, f)

    return

if __name__ == '__main__':
    train_pickle = "train.pickle"
    test_pickle = "test.pickle"
    sent_pickle = "sentiment.pickle"

    main( )
