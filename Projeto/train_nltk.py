import pickle
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer

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

def informative(clf):
    """
        prints on screen the hundred most informative features from Naive Bayes classifier
        writes on file those features
        @clf: Naive Bayes classifier instance
        @rtype: void
    """

    def extract(feature):
        """
            extract feature from string
            @str: string input, ex: 'contains(luv)'
            @ret: string extracted
            @rtype: string
        """

        feature = feature[feature.index("(") + 1: len(feature) - 1]

        return feature.encode("utf-8")

    clf.show_most_informative_features(n = 100)
    features = clf.most_informative_features(n = 100)

    with open(features_path, "wb") as f:
        for feat in features:
            f.write(extract(feat[0]) + b'\n')

    return

def main( ):
    """
        main function
        @rtype: void
    """

    train_data = [ ]
    with open(train_pickle, "rb") as f:
        train_data = pickle.load(f)

    test_data = [ ]
    with open(test_pickle, "rb") as f:
        test_data = pickle.load(f)

    train_data = list(zip(train_data[0], train_data[1]))
    test_data = list(zip(test_data[0], test_data[1]))

    train_data = [(tokenize(t), s) for t, s in train_data]
    test_data = [(tokenize(t), s) for t, s in test_data]

    sentiment = SentimentAnalyzer( )

    words = sentiment.all_words(train_data)
    unigrams = sentiment.unigram_word_feats([mark_negation(w) for w in words], min_freq = 4)
    sentiment.add_feat_extractor(extract_unigram_feats, unigrams = unigrams)

    training_set = sentiment.apply_features(train_data)
    test_set = sentiment.apply_features(test_data)

    trainer = nltk.classify.NaiveBayesClassifier.train

    nbclf = sentiment.train(trainer, training_set)
    sentiment.evaluate(test_set, verbose = True)

    informative(nbclf)

    with open(sent_pickle, "wb") as f:
        pickle.dump(sentiment, f)

    return

if __name__ == '__main__':
    train_pickle = "train.pickle"
    test_pickle = "test.pickle"
    sent_pickle = "sentiment.pickle"

    features_path = "features.txt"

    main( )
