# system
import json


def get_congressional_twitter_data(use_house=True,
                                   use_senate=True,
                                   labels=(0, 1),
                                   verbose=1):

    x = []
    y = []

    label_mapping = {
        'democrat': labels[0],
        'gop': labels[1]
    }

    if use_house:
        with open('./data/twitter/data_house.json') as f:
            house = json.load(f)

        for congressmen in house:
            tweets = congressmen['tweets']
            x.extend([tweet['content'] for tweet in tweets])
            y.extend([label_mapping[congressmen['affiliation']]]*len(tweets))

    if use_senate:
        with open('./data/twitter/data_senate.json') as f:
            senate = json.load(f)

        for congressmen in senate:
            tweets = congressmen['tweets']
            x.extend([tweet['content'] for tweet in tweets])
            y.extend([label_mapping[congressmen['affiliation']]]*len(tweets))

    if verbose > 0:
        print('>>> gathered [ %s ] congressional tweets.' % str(len(x)))
        print('>> [ %s ] are democrats and [ %s ] are GOP' % (len(y) - sum(y), sum(y)))

    return x, y





