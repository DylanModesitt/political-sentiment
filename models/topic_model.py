# system
import sys

# lib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# self
import data.ibc.treeUtil as treeUtil
sys.modules['treeUtil'] = treeUtil
from data.twitter.data import get_congressional_twitter_data
from preprocessing.preprocess import clean_text_documents


def twitter_topic_model(validation_split=0.1,
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
    X, Y = get_congressional_twitter_data()
    X = clean_text_documents(X, twitter=True)
    X = np.array(X)
    Y = np.array(Y)

    no_features = 1000
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(X)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(X)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 10

    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)

    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))

    no_top_words = 10
    display_topics(nmf, tfidf_feature_names, no_top_words)
    print()
    display_topics(lda, tf_feature_names, no_top_words)


def main():
    """
    run a series of nlp tasks on the data we have gathered
    as a baseline in the difficulty of classifying the sentiment
    of political text.

    :return: None.
    """
    twitter_topic_model()


if __name__ == '__main__':
    main()

