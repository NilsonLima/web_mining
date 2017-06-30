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
