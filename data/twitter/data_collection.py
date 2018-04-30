# system
import re
import time
import xlsxwriter
import tweepy
from tweepy import OAuthHandler
import json
import pickle


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class TwitterClient:
    """
    Generic twitter client class to gather, using the
    twitter API,
    """

    class TwitterLists:

        def __init__(self, client):
            self.api = client.api

        def democratic_senators(self):
            return [user.screen_name for user in
                    tweepy.Cursor(self.api.list_members, "TheDemocrats", "senate-democrats").items()]

        def gop_senators(self):
            return [user.screen_name for user in
                    tweepy.Cursor(self.api.list_members, "SenateGOP", "senaterepublicans").items()]

        def democratic_house(self):
            return [user.screen_name for user in
                    tweepy.Cursor(self.api.list_members, "TheDemocrats", "house-democrats").items()]

        def gop_house(self):
            return [user.screen_name for user in
                    tweepy.Cursor(self.api.list_members, "housegop", "house-republicans").items()]

    __COUNT_PER_REQUEST = 200
    __TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self):

        with open("./data/twitter/credentials.txt") as f:
            lines = f.read().splitlines()

        consumer_key, consumer_secret, access_token, access_token_secret = lines
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)
        self.twitter_lists = TwitterClient.TwitterLists(self)

    def get_user_tweets(self,
                        screen_name,
                        additional_meta=None):

        print('>> getting tweets for user [ %s ]' % screen_name)
        user_tweets = []
        oldest = None
        section_count = self.__COUNT_PER_REQUEST
        while section_count == self.__COUNT_PER_REQUEST:
            try:
                print('fetching a new [ %s ] tweets. currently gathered [ %s ].' %
                      (self.__COUNT_PER_REQUEST, len(user_tweets)))

                new_tweets = self.api.user_timeline(screen_name=screen_name,
                                                    count=self.__COUNT_PER_REQUEST,
                                                    max_id=oldest,
                                                    tweet_mode="extended")

                section_count = len(new_tweets)
                user_tweets.extend(new_tweets)
                oldest = (user_tweets[-1].id - 1) if len(user_tweets) > 0 else None
                time.sleep(1)
            except tweepy.error.RateLimitError:
                time_to_sleep = 120
                print('rate limit error. going to sleep for %d\'s. will continue after.' % time_to_sleep)
                time.sleep(time_to_sleep)
                oldest = (user_tweets[-1].id - 1) if len(user_tweets) > 0 else None

        print("total tweets downloaded for [ %s ]: [ %s ]" % (screen_name, len(user_tweets)))

        entries = []
        for tweet in user_tweets:
            if tweet.retweeted or tweet.full_text.startswith("RT "):
                continue

            entries.append({
                'content': tweet.full_text,
                'created_at': tweet.created_at.strftime(self.__TIME_FORMAT),
            })

        data = {**{
            'username': screen_name,
            'tweets': entries
        }, **additional_meta}

        return data


def main():

    api = TwitterClient()

    all_tweets = []

    def save():
        with open('./data/twitter/data.json', 'w') as f:
            json.dump(all_tweets, f)

    print('>>> gathering tweets for all congressmen')
    for screen_name in api.twitter_lists.democratic_senators():
        all_tweets.append(api.get_user_tweets(screen_name, additional_meta={
            'affiliation': 'democrat',
            'assembly': 'senate'
        }))
    save()

    for screen_name in api.twitter_lists.gop_senators():
        all_tweets.append(api.get_user_tweets(screen_name, additional_meta={
            'affiliation': 'gop',
            'assembly': 'senate'
        }))
    save()

    for screen_name in api.twitter_lists.gop_house():
        all_tweets.append(api.get_user_tweets(screen_name, additional_meta={
            'affiliation': 'gop',
            'assembly': 'house'
        }))
    save()

    for screen_name in api.twitter_lists.democratic_house():
        all_tweets.append(api.get_user_tweets(screen_name, additional_meta={
            'affiliation': 'democrat',
            'assembly': 'house'
        }))
    save()


if __name__ == "__main__":
    main()
