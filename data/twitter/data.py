# system
import json
import os

# self
from preprocessing.preprocess import clean_text_documents


def get_congressional_twitter_data(use_house=False,
                                   use_senate=True,
                                   labels=(0, 1),
                                   verbose=1):
    """
    get congressional tweets as a list of samples
    and labels

    :param use_house: use house data
    :param use_senate: use senate data
    :param labels: labels[0] is the dem label, labels[1] is the r label
    :param verbose: logging level
    :return: the lists described above
    """

    x = []
    y = []

    label_mapping = {
        'democrat': labels[0],
        'gop': labels[1]
    }

    if use_house:
        with open('./data/twitter/congress/data_house.json') as f:
            house = json.load(f)

        for congressmen in house:
            tweets = congressmen['tweets']
            x.extend([tweet['content'] for tweet in tweets])
            y.extend([label_mapping[congressmen['affiliation']]]*len(tweets))

    if use_senate:
        with open('./data/twitter/congress/data_senate.json') as f:
            senate = json.load(f)

        for congressmen in senate:
            tweets = congressmen['tweets']
            x.extend([tweet['content'] for tweet in tweets])
            y.extend([label_mapping[congressmen['affiliation']]]*len(tweets))

    if verbose > 0:
        print('>>> gathered [ %s ] congressional tweets.' % str(len(x)))
        print('>> [ %s ] are democrats and [ %s ] are GOP' % (len(y) - sum(y), sum(y)))

    return x, y


def get_political_words(how_many=2000,
                        use_cache_if_present=True,
                        verbose=1):
    """
    get the how_many most common words in the
    tweets of members of congress, except if that
    words is also one of the most common english words.

    Thus, you should have only politically based-words

    :param how_many: how many
    :param use_cache_if_present: whether or not to use the cache
    :param verbose: verbosity
    :return: the words as a set
    """

    x = []

    cache_path = os.path.join('./data/twitter/meta', 'political_words.txt')
    if use_cache_if_present and os.path.isfile(cache_path):
        with open(cache_path) as f:
            return f.read().splitlines()
    else:

        def vprint(*args):
            if verbose > 0:
                print(*args)

        vprint('>> loading data ')
        with open('./data/twitter/congress/data_house.json') as f:
            house = json.load(f)

        for congressmen in house:
            tweets = congressmen['tweets']
            x.extend([tweet['content'] for tweet in tweets])

        with open('./data/twitter/congress/data_senate.json') as f:
            senate = json.load(f)

        for congressmen in senate:
            tweets = congressmen['tweets']
            x.extend([tweet['content'] for tweet in tweets])

        vprint('>> data loaded. cleaning.')

        x = clean_text_documents(x, twitter=True)
        vprint('>> preparing')
        words = [w for s in x for w in s.split()]
        freqs = {}

        for word in words:
            freqs[word] = freqs.get(word, 0) + 1

        with open('./data/twitter/meta/common_words.txt') as f:
            common_words = set(f.read().splitlines())
            for common_word in common_words:
                freqs[common_word] = 0

        political_words = list(zip(*sorted(freqs.items(), key=lambda x: x[1], reverse=True)))[0]
        political_words = list(set(political_words[:(how_many if len(political_words) >=
                                                                 how_many else len(political_words))]))

        with open(cache_path, 'w') as f:
            f.write("\n".join(political_words))

        return political_words













