# system
import sys

# lib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

# self
import data.ibc.treeUtil as treeUtil
sys.modules['treeUtil'] = treeUtil
from data.ibc.data import get_ibc_data
from data.twitter.data import get_congressional_twitter_data
from preprocessing.preprocess import clean_text_documents


def ibc_nlp_classification(validation_split=0.1,
                           shuffle=True):

    """
    fit a series of general nlp classifiers on the
    IBC annotated corpus to compare performance to
    deep learning models

    :param validation_split: the fraction of data to keep
                             for validation
    :param shuffle: whether or not the shuffle the data
    :return: Nothing. results are logged (printed)
    """

    print('>> gathering data \n')
    X, Y = get_ibc_data(use_subsampling=True)
    X = clean_text_documents(X)
    X = np.array(X)
    Y = np.array(Y)

    split = int(validation_split * len(X))

    if shuffle:
        p = np.random.permutation(len(X))
        X = X[p]
        Y = Y[p]

    X_train, X_test = X[split:], X[:split]
    Y_train, Y_test = Y[split:], Y[:split]

    def run_pipeline(pipes):
        text_clf = Pipeline(pipes)
        text_clf.fit(X_train, Y_train)
        predicted = text_clf.predict(X_train)
        print('classifier got [ {} ]% accuracy on training data'.format(np.mean(predicted == Y_train)))
        predicted = text_clf.predict(X_test)
        print('classifier got [ {} ]% accuracy on validation data'.format(np.mean(predicted == Y_test)))

    print('>>> fitting classifiers')

    # SGD Classification
    print('>> SGD Linear Model:')
    run_pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None))
    ])

    # Multinomial Naive Bayes Classification
    print('>> Multinomial Naive Bayes Classifier:')
    run_pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])


def main():
    """
    run a series of nlp tasks on the data we have gathered
    as a baseline in the difficulty of classifying the sentiment
    of political text.

    :return: None.
    """
    ibc_nlp_classification()


if __name__ == '__main__':
    main()

