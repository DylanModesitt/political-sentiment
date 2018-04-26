import re
import time
import xlsxwriter
import tweepy
from tweepy import OAuthHandler

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

    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        with open("credentials.txt") as f:
            lines = [line.rstrip('\n') for line in f]

        consumer_key = lines[0]
        consumer_secret = lines[1]
        access_token = lines[2]
        access_token_secret = lines[3]

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def get_all_tweets(self, screen_name, affiliation):
        #initialize a list to hold all the tweepy Tweets
        alltweets = []

        #make initial request for most recent tweets (200 is the maximum allowed
        new_tweets = []
        while True:
            try:
                new_tweets = self.api.user_timeline(screen_name = screen_name,count=200, tweet_mode="extended")
            except tweepy.error.RateLimitError:
                time.sleep(120)
                continue
            break

        if len(new_tweets) == 0:
            return

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            print("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            while True:
                try:
                    new_tweets = self.api.user_timeline(screen_name = screen_name,count=200,max_id=oldest, tweet_mode="extended")
                except tweepy.error.RateLimitError:
                    time.sleep(120)
                    continue
                break

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print("...%s tweets downloaded so far" % (len(alltweets)))

        # transform the tweepy tweets into a 2D array that will populate the csv
        outtweets = []

        for tweet in alltweets:
            if (tweet.retweeted or tweet.full_text.startswith("RT ")):
                continue

            outtweets.append(tweet.full_text)

        workbook   = xlsxwriter.Workbook('datat/%s/%s.xlsx' % (affiliation, screen_name))
        worksheet = workbook.add_worksheet()

        for i in range(len(outtweets)):
            worksheet.write(i, 0, outtweets[i])

        workbook.close()

    def get_senate_dems(self):
        return [user.screen_name for user in
                tweepy.Cursor(self.api.list_members, "TheDemocrats", "senate-democrats").items()]

    def get_senate_gop(self):
        return [user.screen_name for user in
                tweepy.Cursor(self.api.list_members, "SenateGOP", "senaterepublicans").items()]

    def get_house_dems(self):
        return [user.screen_name for user in
                tweepy.Cursor(self.api.list_members, "TheDemocrats", "house-democrats").items()]

    def get_house_gop(self):
        return [user.screen_name for user in
                tweepy.Cursor(self.api.list_members, "housegop", "house-republicans").items()]


def main():

    api = TwitterClient()

    for screen_name in api.get_senate_dems():
        api.get_all_tweets(screen_name, "dem/senate")

    for screen_name in api.get_senate_gop():
        api.get_all_tweets(screen_name, "gop/senate")


if __name__ == "__main__":
    main()
