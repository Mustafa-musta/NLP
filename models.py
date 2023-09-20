# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import math
import random
stop_words = set(stopwords.words('english'))
some_punkt = ('.', ',', '...', '?', '\'', '!', ':', ';')

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
      
    def get_indexer(self):
        return self.indexer
    def add_vocab(self,ex_words):
    #   indexer = Indexer()

        for ex in ex_words:
            for word in ex.words:
                if word.lower() not in stop_words and word.lower() not in some_punkt: #sET ADDITIONAL CONSIDERATION
                    self.indexer.add_and_get_index(word.lower())
        return self.indexer
    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        features_of_str = np.zeros(self.indexer.__len__())
        for ele in ex_words:
            if self.indexer.contains(ele.lower()):
                features_of_str[self.indexer.index_of(ele.lower())] += 1
        return features_of_str


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
        
    def add_vocab(self,ex_words):
      
        # Generate vocabulary
        for ex in ex_words:
            for i in range(0, len(ex.words) - 1):
                if stop_words.__contains__(ex.words[i]) and stop_words.__contains__(ex.words[i + 1]) or (
                        some_punkt.__contains__(ex.words[i]) or some_punkt.__contains__(ex.words[i + 1])):
                    continue
                bigram = ex.words[i] + ' ' + ex.words[i + 1]
                self.indexer.add_and_get_index(bigram.lower())
        return self.indexer
    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        features_of_str = np.zeros(self.indexer.__len__(), dtype=int)
        for i in range(0, len(ex_words) - 1):
            bigram = ex_words[i] + ' ' + ex_words[i + 1]
            if self.indexer.contains(bigram.lower()):
                index = self.indexer.index_of(bigram.lower())
                features_of_str[index] += 1
        return features_of_str

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def add_vocab(self,ex_words):
        cnt = Counter()
        for ex in ex_words:
            cnt.update(
                word.lower() for word in ex.words if word.lower() not in stop_words and word.lower() not in some_punkt)
        cnt = dict(cnt.most_common(int(cnt.__len__() * 0.5)))
        for keys in cnt.keys():
            self.indexer.add_and_get_index(keys)

        return self.indexer




    def get_indexer(self):
        return self.indexer

    def extract_features(self, ex_words: List[str], add_to_indexer: bool) -> List[int]:
        features_of_str = np.zeros(self.indexer.__len__())
        for ele in ex_words:
            if self.indexer.contains(ele.lower()):
                features_of_str[self.indexer.index_of(ele.lower())] += 1
        return features_of_str


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        features_of_str = self.feat_extractor.extract_features(ex_words, False)
        possibility = np.dot(self.weights, features_of_str)
        #possibility = expo / (1 + expo)
        if possibility > 0.0:
            return 1
        return 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        features_of_str = self.feat_extractor.extract_features(ex_words, False)
        exponantial= math.exp(np.dot(self.weights, features_of_str))
        possibility =  exponantial / (1 + exponantial)
        if possibility > 0.5:
            return 1
        return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    indexer=feat_extractor.add_vocab(train_exs)

    indexer = feat_extractor.get_indexer()
    weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
    learning_rate = .2
    for i in range(20):
        random.shuffle(train_exs)
        for ex in train_exs:
            features_of_str = feat_extractor.extract_features(ex.words, False)
           
            if np.dot(weights, features_of_str)>0:
                possibility=1
            else:
                possibility=0
            
            if (possibility)==(ex.label):
                weights=weights
            elif (possibility==1) and (ex.label==0):
                weights = np.subtract(weights, np.dot(learning_rate, features_of_str))
            elif (possibility==0) and (ex.label==1):
                weights = np.add(weights, np.dot(learning_rate, features_of_str))
                

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer=feat_extractor.add_vocab(train_exs)

    indexer = feat_extractor.get_indexer()
    weights = np.transpose(np.zeros(indexer.__len__(), dtype=int))
    learning_rate = .5
    for i in range(15):
        random.shuffle(train_exs)
        for ex in train_exs:
            features_of_str = feat_extractor.extract_features(ex.words, False)
            exponantial = math.exp(np.dot(weights, features_of_str))
            possibility =  exponantial / (1 + exponantial)
            gradient_of_w = np.dot(ex.label - possibility, features_of_str)
            weights = np.add(weights, np.dot(learning_rate, gradient_of_w))
    return LogisticRegressionClassifier(weights, feat_extractor)



def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model