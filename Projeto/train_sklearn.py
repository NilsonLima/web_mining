import numpy as np

import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import mark_negation

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

def tokenize(tweet):
    """
        tokenize preprocessed tweet and remove stopwords
        @tweet: preprocessed tweet
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
    """

    train_data = [ ]
    with open(train_pickle, "rb") as f:
        train_data = pickle.load(f)

    test_data = [ ]
    with open(test_pickle, "rb") as f:
        test_data = pickle.load(f)

    clf = Pipeline([('vectorizer', CountVectorizer(analyzer = "word",
                                   ngram_range = (1, 2),
                                   tokenizer = tokenize,
                                   #tokenizer = lambda text: mark_negation(tokenize(text)),
                                   max_features = 10000)),
                    ('classifier', MultinomialNB( ))])

    clf.fit(train_data[0], train_data[1])
    predicted = clf.predict(test_data[0])

    metric = metrics.classification_report(test_data[1], predicted, \
                                           target_names = ['negative', 'neutral', 'positive']);

    print(metric)
    print(np.mean(predicted == test_data[1]))

    # params = {"classifier__C": [.01, .1, 1, 10, 100]}
    #
    # gs = GridSearchCV(clf, params, verbose = 2, n_jobs = -1)
    # gs.fit(train_data[0], train_data[1])
    # print(gs.best_estimator_)
    # print(gs.best_score_)

    return

if __name__ == '__main__':
    train_pickle = "train.pickle"
    test_pickle = "test.pickle"
    sent_pickle = "sentiment.pickle"

    main( )
