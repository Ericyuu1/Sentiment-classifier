# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import random
from tqdm import tqdm
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download("stopwords")
#used for tuning
random.seed(1)
#tried and shows dev acc better with stopwords (sad
stopwords= nltk.corpus.stopwords.words("english")
class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer=indexer

    def get_indexer(self):
        return self.indexer

    def add_feature(self,ex_words:List[str]):
        for word in ex_words:
            self.indexer.add_and_get_index(word.lower())

    def extract_features(self,ex_words: List[str], add_to_indexer: bool=False)->List[int]:
        c=Counter()
        for word in ex_words:
            if self.indexer.contains (word.lower()):
                index=self.indexer.index_of(word.lower())
                c.update([index])
        return list(c.items())

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer=indexer

    def get_indexer(self):
        return self.indexer
    
    def add_feature(self,ex_words: List[str]):
        for i in range(len(ex_words)-1):
            #test lowercase
            combo=ex_words[i].lower()+ex_words[i+1].lower()
            self.indexer.add_and_get_index(combo)

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        c=Counter()
        for i in range(len(ex_words)-1):
            #test lowercase
            combo=ex_words[i].lower()+ex_words[i+1].lower()
            if self.indexer.contains(combo):
                index=self.indexer.index_of(combo)
                c.update([index])
        return list(c.items())

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer=indexer

    def get_indexer(self):
        return self.indexer
    '''
    unigram+discard
    def add_feature(self,ex_words:List[str]):
        for word in ex_words:
            if word not in stopwords:
                self.indexer.add_and_get_index(word.lower())
    def extract_features(self,ex_words: List[str], add_to_indexer: bool=False)->List[int]:
        c=Counter()
        for word in ex_words:
            if self.indexer.contains (word.lower()):
                index=self.indexer.index_of(word.lower())
                c.update([index])
        return list(c.items())
    '''
    '''
    TRIGRAM
    def add_feature(self,ex_words: List[str]):
        for i in range(len(ex_words)-2):
            #test lowercase
            combo=ex_words[i].lower()+ex_words[i+1].lower()+ex_words[i+2]
            self.indexer.add_and_get_index(combo)

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        c=Counter()
        for i in range(len(ex_words)-2):
            #test lowercase
            combo=ex_words[i].lower()+ex_words[i+1].lower()+ex_words[i+2]
            if self.indexer.contains(combo):
                index=self.indexer.index_of(combo)
                c.update([index])
        return list(c.items())
    '''
    def add_feature(self,ex_words: List[str]):
        for i in range(len(ex_words)-1):
            #test lowercase
            if ex_words[i] not in string.punctuation or ex_words[i+1] not in string.punctuation:
                combo=ex_words[i].lower()+ex_words[i+1].lower()
                self.indexer.add_and_get_index(combo)

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        c=Counter()
        for i in range(len(ex_words)-1):
            #test lowercase
            combo=ex_words[i].lower()+ex_words[i+1].lower()
            if self.indexer.contains(combo):
                index=self.indexer.index_of(combo)
                c.update([index])
        return list(c.items())

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex_words: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    ##
    def __init__(self,feat_extractor):
        self.feat_extractor=feat_extractor
        self.indexer=self.feat_extractor.get_indexer()
        self.wordsize=self.indexer.__len__()
        self.w=np.zeros((self.wordsize,))
        self.dict={}
    
    def get_feature(self, ex_words: List[str])->List[int]:
        #convert list to str
        ex_str = "".join(ex_words)
        feature=[]
        if ex_str not in self.dict:
            feature = self.feat_extractor.extract_features(ex_words)
            self.dict[str] = feature
        else:
            feature= self.dict[ex_str]
        return feature
        
    def predict(self,ex_words: List[str])->int:
        feat_num=self.get_feature(ex_words)
        sum=0.
        for key,val in feat_num:
            sum+=self.w[key]*val
        p=sigmoid(sum)
        if p>0.5:
            y_pred=1
        else: y_pred=0
        return y_pred

    def update(self, ex_words: List[str],y,y_pred,step):
        feat_num=self.get_feature(ex_words)
        sum=0.
        for key,val in feat_num:
            sum+=self.w[key]*val
        p=sigmoid(sum)
        
        """
        loss = -y*log(p)-(1-y)*log(1-p)
        d(loss)/dw = (p-y)*fx
        """
        for key,val in feat_num:
            #SGD
            self.w[key]=self.w[key]-step*((p-y)*val)
    #loss = -y*log(p)-(1-y)*log(1-p)
    #d(loss)/dw = (p-y)*fx
    """
    FOR PLOT USE
    def loss_avg(self,train_exs):
        sum=0
        for wordss in train_exs:
            y=wordss.label
            feat_num=self.get_feature(wordss.words)
            x=0
            for key,val in feat_num:
                x +=self.w[key]*val
            p=sigmoid(x)
            loss=-y*np.log(p)-(1-y)*np.log(1-p)
            sum+=loss
        return sum/float(len(train_exs))
    """
def sigmoid(x):
    return 1./(1.+np.exp(-x))

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    raise Exception("Must be implemented")


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    for wlst in train_exs: 
        feat_extractor.add_feature(wlst.words)

    model=LogisticRegressionClassifier(feat_extractor)
    #set epochs number (has tried several)
    epochs=20

    step=0.07
    for i in tqdm(range(epochs)):
        # get different training set
        random.shuffle(train_exs)
        for wlst in train_exs:
            y=wlst.label
            y_pred=model.predict(wlst.words)
            model.update(wlst.words,y,y_pred,step)
    '''
    list of loss has been deleted.
    plt.plot(x,loss,label="Step=0.5")
    plt.plot(x,loss01,color="red",label="Step=0.1")
    plt.plot(x,loss005,color="green",label="Step=0.05")
    plt.legend(loc='upper right')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.xticks(np.linspace(0,19,20))
    plt.show()
    '''
    '''
    list of acc has been deleted.
    plt.plot(x,acc,label="Step=0.5")
    plt.plot(x,acc01,color="red",label="Step=0.1")
    plt.plot(x,acc005,color="green",label="Step=0.05")
    plt.legend(loc='upper right')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.xticks(np.linspace(0,19,20))
    plt.show()
    '''
    return model

def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
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